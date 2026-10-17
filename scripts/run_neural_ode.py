import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Post-hoc Neural ODE with common baseline anchors and recorded RK4 dynamics")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="configs/remaining_hidden_dynamics.json")
    parser.add_argument("--device", required=True)
    parser.add_argument("--datasets", nargs="+")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mosd_hidden_dynamics.extended import fit_pca, load_extended_group, project, series_metrics
    from mosd_hidden_dynamics.neural_ode import field_diagnostics, fit_ode, load_ode, rollout_ode, save_ode
    from mosd_hidden_dynamics.study import aggregate_results, write_rows
    from mosd_hidden_dynamics.utils import save_json, source_record
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(args.run_dir).resolve()
    cfg = json.loads((root / "config_requested.json").read_text())
    extra = json.loads(Path(args.config).read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    output = root / "neural_ode"
    output.mkdir(exist_ok=True)
    requested = output / "config_requested.json"
    if requested.exists() and json.loads(requested.read_text()) != extra:
        raise ValueError("Neural ODE settings changed")
    save_json(requested, extra)
    source = source_record(repo)
    for dataset in cfg["datasets"]:
        name = dataset["name"]
        if dataset["family"] == "control" or args.datasets and name not in args.datasets:
            continue
        for protocol in ("in_series", "out_of_series"):
            directory = output / (name + "_" + protocol)
            if (directory / "complete.json").exists():
                continue
            directory.mkdir(exist_ok=True)
            start = time.perf_counter()
            group, anchors, digest = load_extended_group(cfg, registry, root, name, protocol,
                                                           extra["history_order"], extra["ode"]["segment_steps"])
            train, val = group["train"], group["val"]
            write_rows(directory / "anchors.csv", anchors)
            rows, failures = [], []
            common = {"dataset": name, "protocol": protocol, "condition": "pretrained", "family": dataset["family"]}
            for k in cfg["horizons"]:
                for method, prediction in (("persistence", val["x"]),
                                           ("train_mean", np.broadcast_to(train[f"y{k}"].mean(0), val[f"y{k}"].shape))):
                    rows.extend({**common, "model": method, "horizon": k, "seed": None, **r}
                                for r in series_metrics(val[f"y{k}"], prediction, train[f"y{k}"].mean(0), val["ids"]))
            pca = fit_pca(train["x"], .95, args.device)
            for seed in extra["seeds"]:
                path = directory / f"ode_seed{seed}.pt"
                if path.exists():
                    fit = load_ode(path, args.device)
                else:
                    fit = fit_ode(train, val, extra["ode"], seed, args.device)
                    save_ode(path, fit)
                print(name, protocol, "ODE seed", seed, "best epoch", fit["best_epoch"],
                      "training seconds", round(fit["seconds"]), flush=True)
                predictions = rollout_ode(fit, val["x"], cfg["horizons"])
                for k, prediction in predictions.items():
                    rows.extend({**common, "model": "neural_ode", "horizon": k, "seed": seed, **r}
                                for r in series_metrics(val[f"y{k}"], prediction, train[f"y{k}"].mean(0), val["ids"]))
                half = rollout_ode(fit, val["x"], [1, 4], substeps=2)
                save_json(directory / f"integration_seed{seed}.json",
                          {str(k): {"relative_l2_change_half_step": float(np.linalg.norm(half[k] - predictions[k]) / np.linalg.norm(half[k])),
                                    "relative_l2_error": float(np.linalg.norm(half[k] - val[f"y{k}"]) / np.linalg.norm(val[f"y{k}"]))}
                           for k in (1, 4)})
                if protocol == "out_of_series":
                    long_cfg = copy.deepcopy(cfg)
                    long_cfg["horizons"] = [1, 4, 16, 64, 128, extra["ode"]["rollout_steps"]]
                    long_group, long_anchors, long_digest = load_extended_group(
                        long_cfg, registry, root, name, protocol, extra["history_order"], extra["ode"]["segment_steps"])
                    ids = long_group["val"]["ids"]
                    rng = np.random.default_rng(seed)
                    selections = []
                    for series in np.unique(ids):
                        choices = np.flatnonzero(ids == series)
                        selections.extend(rng.choice(choices, min(len(choices), int(np.ceil(50 / len(np.unique(ids))))), replace=False))
                    indices = np.asarray(selections[:extra["ode"]["rollout_starts"]])
                    long_pred = rollout_ode(fit, long_group["val"]["x"][indices], long_cfg["horizons"])
                    terminal = long_pred[extra["ode"]["rollout_steps"]]
                    centre = terminal.mean(0)
                    relative = np.linalg.norm(terminal - centre, axis=1) / max(np.linalg.norm(centre), 1e-12)
                    long_rows = []
                    for k, prediction in long_pred.items():
                        long_rows.extend({**common, "model": "neural_ode", "horizon": k, "seed": seed, **r} for r in
                                         series_metrics(long_group["val"][f"y{k}"][indices], prediction,
                                                        long_group["train"][f"y{k}"].mean(0), ids[indices]))
                    save_json(directory / f"long_seed{seed}.json", {"anchor_sha256": long_digest, "rows": long_rows,
                              "starts": len(indices), "training_segment_steps": extra["ode"]["segment_steps"],
                              "rollout_steps": extra["ode"]["rollout_steps"],
                              "fraction_within_0_1_relative_l2_of_terminal_barycentre": float(np.mean(relative < .1))})
                    np.savez_compressed(directory / f"trajectories_seed{seed}.npz",
                                        endpoints=long_cfg["horizons"],
                                        trajectories=np.stack([project(pca, long_pred[k])[..., :2] for k in long_cfg["horizons"]], axis=1),
                                        observed=project(pca, val["x"])[..., :2])
                    del long_group
                diagnostics, roots, spectra = field_diagnostics(fit, train["x"], val["x"], extra["ode"])
                save_json(directory / f"roots_seed{seed}.json", diagnostics)
                np.savez_compressed(directory / f"roots_seed{seed}.npz", roots=roots, eigenvalues=spectra)
                save_json(directory / f"fit_seed{seed}.json", {key: fit[key] for key in
                          ("best_epoch", "history", "seconds", "parameters", "seed")})
                save_json(directory / "metrics.json", rows)
                del fit
            write_rows(directory / "metrics.csv", rows)
            save_json(directory / "run_metadata.json", {"source": source, "anchor_sha256": digest,
                      "integrator": "fixed-step classical RK4; half-step sensitivity reported",
                      "selection": "macro validation R2 at k=4; three seeds retained",
                      "test_evaluated": False})
            save_json(directory / "complete.json", {"anchor_sha256": digest, "seconds": time.perf_counter() - start,
                      "test_evaluated": False, "failures": failures})
            print("DONE ODE", name, protocol, flush=True)
            del group, train, val
            torch.cuda.empty_cache()
    all_rows = []
    for path in output.glob("*/metrics.json"):
        if (path.parent / "complete.json").exists():
            all_rows.extend(json.loads(path.read_text()))
    if all_rows:
        summaries = aggregate_results(all_rows, cfg["bootstrap_samples"], cfg["seed"])
        save_json(output / "summary.json", summaries)
        write_rows(output / "summary.csv", summaries)


if __name__ == "__main__":
    main()
