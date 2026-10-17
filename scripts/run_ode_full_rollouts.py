import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Frozen ODE and ridge rollouts on identical long-horizon anchors")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mosd_hidden_dynamics.extended import fit_pca, load_extended_group, project, series_metrics
    from mosd_hidden_dynamics.neural_ode import advance, load_ode
    from mosd_hidden_dynamics.study import aggregate_results, write_rows
    from mosd_hidden_dynamics.study_models import load_ridge
    from mosd_hidden_dynamics.utils import save_json, source_record
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(args.run_dir).resolve()
    cfg = json.loads((root / "config_requested.json").read_text())
    settings = json.loads((root / "residual_training/config_requested.json").read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    output = root / "ode_full_rollouts"
    output.mkdir(exist_ok=True)
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
        long_cfg = copy.deepcopy(cfg)
        long_cfg["horizons"] = settings["horizons"]
        group, anchors, digest = load_extended_group(long_cfg, registry, root, name, "out_of_series", order=8, segment=8)
        residual_meta = json.loads((root / "residual_training" / name / "run_metadata.json").read_text())
        if digest != residual_meta["evaluation_anchor_sha256"]:
            raise ValueError("ODE and residual long-horizon anchors differ")
        train, val = group["train"], group["val"]
        write_rows(directory / "anchors.csv", anchors)
        rng = np.random.default_rng(cfg["seed"])
        chosen = []
        series_ids = np.unique(val["ids"])
        for series in series_ids:
            candidates = np.flatnonzero(val["ids"] == series)
            count = min(len(candidates), int(np.ceil(50 / len(series_ids))))
            chosen.extend(rng.choice(candidates, count, replace=False))
        chosen = np.asarray(chosen[:50], dtype=int)
        val_anchors = [row for row in anchors if row["split"] == "val"]
        if [row["series_id"] for row in val_anchors] != val["ids"].tolist():
            raise ValueError("Anchor order does not match observations")
        pca = fit_pca(train["x"], .95, args.device)
        if pca["rank"] < 2:
            raise ValueError("Two trajectory coordinates require at least two principal components")
        mean = pca["mean"].cpu().numpy()
        vectors = pca["vectors"][:, :2].cpu().numpy()
        trajectories = {"initial_indices": chosen, "pca_mean": mean, "pca_vectors": vectors,
                        "series_ids": val["ids"][chosen],
                        "source_ends": np.asarray([val_anchors[i]["source_end"] for i in chosen])}
        truth = np.empty((len(chosen), max(settings["horizons"]) + 1, 2))
        for series in np.unique(val["ids"][chosen]):
            with np.load(root / "features/pretrained" / (str(series) + ".npz")) as data:
                states, endpoints = data["states"][:, 0].astype(np.float32), data["raw_endpoints"]
            for j in np.flatnonzero(val["ids"][chosen] == series):
                end = val_anchors[chosen[j]]["source_end"]
                index = int(np.searchsorted(endpoints, end))
                if endpoints[index] != end:
                    raise ValueError("Trajectory source endpoint missing")
                segment = states[index:index + truth.shape[1]]
                if len(segment) != truth.shape[1]:
                    raise ValueError("Incomplete observed trajectory")
                truth[j] = (segment - mean) @ vectors
        trajectories["observed"] = truth
        rows, concentration = [], []
        common = {"dataset": name, "family": dataset["family"], "protocol": "out_of_series", "condition": "pretrained"}

        def record(method, k, prediction, seed=None):
            rows.extend({**common, "model": method, "horizon": k, "seed": seed, **row}
                        for row in series_metrics(val[f"y{k}"], prediction, train[f"y{k}"].mean(0), val["ids"]))

        for k in settings["horizons"]:
            record("persistence", k, val["x"])
            record("train_mean", k, np.broadcast_to(train[f"y{k}"].mean(0), val[f"y{k}"].shape))
        fits = [("ridge_original", None, load_ridge(root / "remaining_baselines" / f"{name}_out_of_series/ridge_direct_k1.npz", args.device))]
        fits.extend(("neural_ode", seed, load_ode(root / "neural_ode" / f"{name}_out_of_series/ode_seed{seed}.pt", args.device))
                    for seed in settings["seeds"])
        for method, seed, fit in fits:
            path = np.empty_like(truth)
            path[:, 0] = (val["x"][chosen] - mean) @ vectors
            with torch.inference_mode():
                if method == "neural_ode":
                    state = torch.as_tensor((val["x"] - fit["mean"]) / fit["scale"], device=args.device, dtype=torch.float32)
                    offset = torch.as_tensor(fit["mean"], device=args.device)
                    scale = torch.as_tensor(fit["scale"], device=args.device)
                else:
                    state = torch.as_tensor(val["x"], device=args.device, dtype=torch.float64)
                projected_mean = torch.as_tensor(mean, device=args.device, dtype=torch.float64)
                projected_vectors = torch.as_tensor(vectors, device=args.device, dtype=torch.float64)
                for k in range(1, path.shape[1]):
                    if method == "neural_ode":
                        state = advance(fit["model"], state, substeps=fit["config"]["rk4_substeps"])
                        raw = state.to(torch.float64) * scale + offset
                    else:
                        state = ((state - fit["mean"]) / fit["scale"]) @ fit["coef"].T + fit["intercept"]
                        raw = state
                    path[:, k] = ((raw[chosen] - projected_mean) @ projected_vectors).cpu().numpy()
                    if k in settings["horizons"]:
                        prediction = raw.cpu().numpy()
                        if not np.isfinite(prediction).all():
                            raise FloatingPointError(f"Nonfinite {method} rollout at step {k}")
                        record(method, k, prediction, seed)
                terminal = raw[chosen].cpu().numpy()
            centre = terminal.mean(0)
            relative = np.linalg.norm(terminal - centre, axis=1) / max(np.linalg.norm(centre), 1e-12)
            concentration.append({"model": method, "seed": seed, "starts": len(chosen),
                "fraction_within_0_1_relative_l2_of_terminal_barycentre": float(np.mean(relative < .1))})
            trajectories[method + ("" if seed is None else f"_seed{seed}")] = path
            print("FULL ROLLOUT", name, method, seed, len(val["x"]), "anchors", flush=True)
        np.savez_compressed(directory / "trajectories.npz", **trajectories)
        save_json(directory / "metrics.json", rows)
        write_rows(directory / "metrics.csv", rows)
        save_json(directory / "concentration.json", concentration)
        save_json(directory / "run_metadata.json", {"source": source, "anchor_sha256": digest,
            "trajectory_selection_seed": cfg["seed"], "trajectory_starts": len(chosen),
            "metric_anchors": len(val["x"]), "training": "frozen k<=4 ODE and one-step ridge checkpoints; no refitting",
            "pca": "training states only; two leading components", "test_evaluated": False})
        save_json(directory / "complete.json", {"seconds": time.perf_counter() - started, "test_evaluated": False})
        del group, train, val, fits
        torch.cuda.empty_cache()
    rows = []
    for path in output.glob("*/metrics.json"):
        if (path.parent / "complete.json").exists():
            rows.extend(json.loads(path.read_text()))
    if rows:
        summary = aggregate_results(rows, cfg["bootstrap_samples"], cfg["seed"])
        save_json(output / "summary.json", summary)
        write_rows(output / "summary.csv", summary)


if __name__ == "__main__":
    main()
