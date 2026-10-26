import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Matched validation-only autonomous transition comparison")
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", choices=("Lorenz", "Rossler", "electricity"), required=True)
    parser.add_argument("--config", default="configs/consistent_hidden_dynamics.json")
    parser.add_argument("--device", required=True)
    parser.add_argument("--stop-file", required=True)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    import numpy as np
    import torch
    from mosd_hidden_dynamics.consistent import (fit_transition, fixed_starts, load_transition, path_statistics,
        projected_paths, save_transition, score_series, scoring_space, transition)
    from mosd_hidden_dynamics.extended import load_extended_group, series_metrics
    from mosd_hidden_dynamics.study_models import fit_ridge, prepare_ridge, rollout_ridge, save_ridge
    from mosd_hidden_dynamics.utils import environment_record, save_json, source_record
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads(Path(args.config).read_text())
    root, out = Path(args.reference_run).resolve(), Path(args.output_dir).resolve() / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    assert not out.is_relative_to(repo)
    requested = out / "config_requested.json"
    if requested.exists():
        assert json.loads(requested.read_text()) == cfg
    save_json(requested, cfg)
    source = source_record(repo)
    save_json(out / "environment.json", environment_record())
    base = json.loads((root / "config_requested.json").read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    short, train_anchors, digest = load_extended_group(base, registry, root, args.dataset, "out_of_series", order=8, segment=4)
    long_cfg = copy.deepcopy(base)
    long_cfg["horizons"] = cfg["horizons"]
    long, anchors, long_digest = load_extended_group(long_cfg, registry, root, args.dataset, "out_of_series", order=8, segment=4)
    train, val, future = short["train"], short["val"], long["val"]
    val_anchors = [r for r in anchors if r["split"] == "val"]
    assert [r["series_id"] for r in val_anchors] == future["ids"].tolist()
    save_json(out / "training_and_selection_anchors.json", train_anchors)
    save_json(out / "report_anchors.json", val_anchors)
    space = scoring_space(train["x"], args.device)
    np.savez_compressed(out / "scoring_space.npz", **space)
    selected = fixed_starts(future["ids"], cfg["trajectory_starts_per_series"])
    vectors = space["vectors"][:, :cfg["projection_dimensions"]]
    truth = projected_paths(root, val_anchors, selected, vectors, space["mean"], max(cfg["horizons"]))
    np.savez_compressed(out / "observed_paths.npz", paths=truth, indices=selected, ids=future["ids"][selected])
    train_radius = float(np.quantile(np.linalg.norm(train["x"] - space["mean"], axis=1), .99))
    references = {k: long["train"][f"y{k}"].mean(0) for k in cfg["horizons"]}
    np.savez_compressed(out / "report_targets.npz", ids=future["ids"], source_ends=np.asarray([r["source_end"] for r in val_anchors]),
                        **{f"y{k}": future[f"y{k}"] for k in cfg["horizons"]},
                        **{f"reference{k}": references[k] for k in cfg["horizons"]})
    save_json(out / "run_metadata.json", {"source": source, "training_anchor_sha256": digest,
        "report_anchor_sha256": long_digest, "test_evaluated": False,
        "models": ["ridge", "mlp", "residual", "ode"], "projection_dimensions": vectors.shape[1],
        "normalization": "common short-manifest training input statistics; scoring references from long-manifest training targets",
        "path_statistics": "finite projected validation trajectories; no invariant-measure or recovered-attractor claim"})

    def record(name, seed, predictions, path, bounded, extra=None):
        rows = []
        for k, prediction in predictions.items():
            if not np.isfinite(prediction).all():
                rows.append({"dataset": args.dataset, "model": name, "seed": seed, "horizon": k, "status": "nonfinite_prediction"})
                continue
            rows.extend({"dataset": args.dataset, "model": name, "seed": seed, "horizon": k, "status": "complete", **r}
                        for r in score_series(future[f"y{k}"], prediction, references[k], future["ids"], space))
        save_json(out / f"{name}_{seed}_metrics.json", rows)
        np.savez_compressed(out / f"{name}_{seed}_predictions.npz", **{f"k{k}": value for k, value in predictions.items()})
        np.savez_compressed(out / f"{name}_{seed}_paths.npz", paths=path)
        stats = path_statistics(path, truth, space["pc_scale"][:vectors.shape[1]]) if np.isfinite(path).all() else {"status": "nonfinite_path"}
        save_json(out / f"{name}_{seed}_path_metrics.json", {**stats, "bounded_fraction": float(bounded.mean()),
            "bounded_rule": "all 256 predicted states within ten times training 99th-percentile centered radius", **(extra or {})})

    for name in ("train_mean", "persistence"):
        predictions = {k: np.broadcast_to(references[k], future[f"y{k}"].shape) if name == "train_mean" else future["x"] for k in cfg["horizons"]}
        # The mean baseline is horizon-specific for point scores; its displayed path is the training state mean.
        start = ((future["x"][selected] if name == "persistence" else np.broadcast_to(space["mean"], future["x"][selected].shape)) - space["mean"]) @ vectors
        path = np.repeat(start[:, None], max(cfg["horizons"]) + 1, axis=1)
        record(name, 0, predictions, path, np.ones(len(selected), dtype=bool))
    prepared = prepare_ridge(train["x"], args.device)
    best, choices = None, []
    for alpha in cfg["ridge_alphas"]:
        fit = fit_ridge(prepared, train["y1"], alpha)
        predicted = rollout_ridge(fit, val["x"], [4])[4]
        value = float(np.mean([r["r2"] for r in series_metrics(val["y4"], predicted, train["y4"].mean(0), val["ids"])]))
        choices.append({"alpha": alpha, "validation_macro_r2_k4": value})
        if best is None or (value, -alpha) > best[:2]:
            best = value, -alpha, fit
    save_json(out / "ridge_selection.json", choices)
    fit = best[2]
    save_ridge(out / "ridge.npz", fit)
    state = torch.as_tensor(future["x"], device=args.device, dtype=torch.float64)
    ridge_paths = [((future["x"][selected] - space["mean"]) @ vectors)]
    predictions, bounded = {}, np.ones(len(selected), dtype=bool)
    with torch.inference_mode():
        for k in range(1, max(cfg["horizons"]) + 1):
            state = ((state - fit["mean"]) / fit["scale"]) @ fit["coef"].T + fit["intercept"]
            picked = state[selected].cpu().numpy()
            ridge_paths.append((picked - space["mean"]) @ vectors)
            bounded &= np.linalg.norm(picked - space["mean"], axis=1) <= cfg["bounded_radius_multiplier"] * train_radius
            if k in cfg["horizons"]:
                predictions[k] = state.cpu().numpy()
    record("ridge", 0, predictions, np.stack(ridge_paths, axis=1), bounded)
    print(args.dataset, "ridge complete", flush=True)
    del prepared, fit, state

    for method in ("mlp", "residual", "ode"):
        for seed in cfg["seeds"]:
            if Path(args.stop_file).exists():
                raise InterruptedError("Stop requested")
            complete_path = out / f"{method}_{seed}_complete.json"
            if complete_path.exists():
                continue
            checkpoint = out / f"{method}_{seed}.pt"
            if checkpoint.exists():
                fit = load_transition(checkpoint, args.device)
            else:
                fit = fit_transition(train, val, cfg["training"], method, seed, args.device, Path(args.stop_file))
                save_transition(checkpoint, fit)
            print(args.dataset, method, seed, "fit minutes", round(fit["seconds"] / 60, 2), "epoch", fit["best_epoch"], flush=True)
            save_json(out / f"{method}_{seed}_training.json", {k: fit[k] for k in ("seconds", "best_epoch", "parameters", "history")})
            predictions, paths, bounded = {}, [((future["x"][selected] - space["mean"]) @ vectors)], np.ones(len(selected), dtype=bool)
            with torch.inference_mode():
                state = torch.as_tensor((future["x"] - fit["mean"]) / fit["scale"], device=args.device, dtype=torch.float32)
                for k in range(1, max(cfg["horizons"]) + 1):
                    if Path(args.stop_file).exists():
                        raise InterruptedError("Stop requested")
                    state = transition(fit["model"], state, method, cfg["training"]["rk4_substeps"])
                    picked = state[selected].cpu().numpy().astype(np.float64) * fit["scale"] + fit["mean"]
                    paths.append((picked - space["mean"]) @ vectors)
                    bounded &= np.linalg.norm(picked - space["mean"], axis=1) <= cfg["bounded_radius_multiplier"] * train_radius
                    if k in cfg["horizons"]:
                        predictions[k] = state.cpu().numpy().astype(np.float64) * fit["scale"] + fit["mean"]
            record(method, seed, predictions, np.stack(paths, axis=1), bounded)
            if method == "ode":
                finer = {}
                for substeps in (8, 16):
                    with torch.inference_mode():
                        state = torch.as_tensor((val["x"] - fit["mean"]) / fit["scale"], device=args.device, dtype=torch.float32)
                        for _ in range(4):
                            state = transition(fit["model"], state, method, substeps)
                        finer[substeps] = state.cpu().numpy().astype(np.float64) * fit["scale"] + fit["mean"]
                save_json(out / f"ode_{seed}_resolution.json", {"relative_prediction_difference_8_vs_16_k4":
                    float(np.linalg.norm(finer[8] - finer[16]) / np.linalg.norm(finer[16])),
                    "substeps16_entity_metrics": score_series(val["y4"], finer[16], train["y4"].mean(0), val["ids"], space)})
            save_json(complete_path, {"status": "complete", "test_evaluated": False})
            print(args.dataset, method, seed, "rollout complete", flush=True)
            del fit, state
    save_json(out / "complete.json", {"test_evaluated": False, "status": "complete"})


if __name__ == "__main__":
    main()
