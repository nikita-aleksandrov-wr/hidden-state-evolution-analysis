import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Matched residual maps with one-/multi-step training and spectral-radius controls")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="configs/residual_training.json")
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mosd_hidden_dynamics.extended import load_extended_group, series_metrics
    from mosd_hidden_dynamics.neural_ode import fit_ode, load_ode, rollout_ode, save_ode
    from mosd_hidden_dynamics.study import aggregate_results, write_rows
    from mosd_hidden_dynamics.study_models import load_ridge, rollout_ridge, save_ridge
    from mosd_hidden_dynamics.utils import save_json, source_record
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(args.run_dir).resolve()
    cfg = json.loads((root / "config_requested.json").read_text())
    settings = json.loads(Path(args.config).read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    output = root / "residual_training"
    output.mkdir(exist_ok=True)
    requested = output / "config_requested.json"
    if requested.exists() and json.loads(requested.read_text()) != settings:
        raise ValueError("Residual-training config changed")
    save_json(requested, settings)
    source = source_record(repo)
    for dataset in cfg["datasets"]:
        if dataset["family"] == "control":
            continue
        name = dataset["name"]
        directory = output / name
        if (directory / "complete.json").exists():
            continue
        directory.mkdir(exist_ok=True)
        started = time.perf_counter()
        group, anchors, digest = load_extended_group(cfg, registry, root, name, "out_of_series", order=8, segment=8)
        long_cfg = copy.deepcopy(cfg)
        long_cfg["horizons"] = settings["horizons"]
        long_group, long_anchors, long_digest = load_extended_group(long_cfg, registry, root, name,
                                                                    "out_of_series", order=8, segment=8)
        train, val = group["train"], long_group["val"]
        write_rows(directory / "training_anchors.csv", anchors)
        write_rows(directory / "evaluation_anchors.csv", long_anchors)
        common = {"dataset": name, "family": dataset["family"], "protocol": "out_of_series", "condition": "pretrained"}
        rows = []
        def record(method, k, prediction, seed=None):
            rows.extend({**common, "model": method, "horizon": k, "seed": seed, **r} for r in
                        series_metrics(val[f"y{k}"], prediction, long_group["train"][f"y{k}"].mean(0), val["ids"]))
        for k in settings["horizons"]:
            record("persistence", k, val["x"])
            record("train_mean", k, np.broadcast_to(long_group["train"][f"y{k}"].mean(0), val[f"y{k}"].shape))
        for steps in settings["training_steps"]:
            model_cfg = dict(settings["model"], segment_steps=steps)
            for seed in settings["seeds"]:
                path = directory / f"residual_steps{steps}_seed{seed}.pt"
                if path.exists():
                    fit = load_ode(path, args.device)
                else:
                    fit = fit_ode(train, group["val"], model_cfg, seed, args.device)
                    save_ode(path, fit)
                for k, prediction in rollout_ode(fit, val["x"], settings["horizons"]).items():
                    record(f"residual_training{steps}", k, prediction, seed)
                save_json(directory / f"fit_steps{steps}_seed{seed}.json",
                          {key: fit[key] for key in ("seconds", "history", "seed", "best_epoch", "parameters")})
                save_json(directory / "metrics.json", rows)
                print(name, "training steps", steps, "seed", seed, "epoch", fit["best_epoch"],
                      "seconds", round(fit["seconds"]), flush=True)
                del fit
        path = root / "remaining_baselines" / f"{name}_out_of_series" / "ridge_direct_k1.npz"
        complete = path.parent / "complete.json"
        if not complete.exists():
            # The independent baseline job must finish its own case before a frozen map is read.
            waited = time.monotonic()
            while not complete.exists():
                if time.monotonic() - waited > 1800:
                    raise TimeoutError("Common-anchor ridge case did not finish")
                time.sleep(2)
        fit = load_ridge(path, args.device)
        radius = float(torch.linalg.eigvals(fit["coef"] / fit["scale"][None]).abs().max())
        linear = [("ridge_original", fit)]
        for limit in settings["spectral_radius_limits"]:
            modified = {key: value.clone() for key, value in fit.items()}
            modified["coef"] *= min(1, limit / max(radius, 1e-12))
            linear.append((f"ridge_radius_{limit}", modified))
            save_ridge(directory / f"ridge_radius_{limit}.npz", modified)
        for method, model in linear:
            for k, prediction in rollout_ridge(model, val["x"], settings["horizons"]).items():
                record(method, k, prediction)
        save_json(directory / "metrics.json", rows)
        write_rows(directory / "metrics.csv", rows)
        save_json(directory / "run_metadata.json", {"source": source, "training_anchor_sha256": digest,
                  "evaluation_anchor_sha256": long_digest, "original_ridge_radius": radius,
                  "selection": "same k=4 macro validation R2 for every training objective; all seeds retained",
                  "test_evaluated": False})
        save_json(directory / "complete.json", {"seconds": time.perf_counter() - started, "test_evaluated": False})
        del group, long_group, train, val, fit
        torch.cuda.empty_cache()
    rows = []
    for path in output.glob("*/metrics.json"):
        if (path.parent / "complete.json").exists():
            rows.extend(json.loads(path.read_text()))
    if rows:
        summaries = aggregate_results(rows, cfg["bootstrap_samples"], cfg["seed"])
        save_json(output / "summary.json", summaries)
        write_rows(output / "summary.csv", summaries)


if __name__ == "__main__":
    main()
