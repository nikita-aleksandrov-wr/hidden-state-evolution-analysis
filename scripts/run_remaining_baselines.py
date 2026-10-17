import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Common-anchor PCA, DMD/VAR, controlled and matched nonlinear baselines")
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
    from mosd_hidden_dynamics.extended import (coordinate_scores, dmd_rollout, fit_dmd, fit_pca, fit_var,
        load_extended_group, matched_mlp_config, series_metrics, var_rollout)
    from mosd_hidden_dynamics.study import aggregate_results, depth_comparison, write_rows
    from mosd_hidden_dynamics.study_models import (fit_mlp, predict_mlp, predict_ridge, prepare_ridge,
        save_ridge, select_ridge, rollout_ridge)
    from mosd_hidden_dynamics.utils import save_json, source_record
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(args.run_dir).resolve()
    cfg = json.loads((root / "config_requested.json").read_text())
    base = json.loads((root / "backbone_config.json").read_text())
    extra = json.loads(Path(args.config).read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    output = root / "remaining_baselines"
    output.mkdir(exist_ok=True)
    requested_path = output / "config_requested.json"
    if requested_path.exists() and json.loads(requested_path.read_text()) != extra:
        raise ValueError("Remaining-study settings changed")
    save_json(requested_path, extra)
    source = source_record(repo)
    for dataset in cfg["datasets"]:
        name = dataset["name"]
        if args.datasets and name not in args.datasets:
            continue
        for protocol in ("in_series", "out_of_series"):
            directory = output / (name + "_" + protocol)
            if (directory / "complete.json").exists():
                continue
            directory.mkdir(exist_ok=True)
            start = time.perf_counter()
            group, anchors, anchor_hash = load_extended_group(cfg, registry, root, name, protocol,
                                                               extra["history_order"], extra["ode"]["segment_steps"])
            train, val = group["train"], group["val"]
            write_rows(directory / "anchors.csv", anchors)
            save_json(directory / "run_metadata.json", {"source": source, "anchor_sha256": anchor_hash,
                      "test_evaluated": False, "selection": "macro validation R2; all seeds retained"})
            rows, selections, dimensions = [], {}, []
            common = {"dataset": name, "protocol": protocol, "condition": "pretrained", "family": dataset["family"]}

            def record(model, horizon, prediction, seed=None):
                target = val[f"y{horizon}"]
                reference = train[f"y{horizon}"].mean(0)
                rows.extend({**common, "model": model, "horizon": horizon, "seed": seed,
                             **r} for r in series_metrics(target, prediction, reference, val["ids"]))
                dimensions.append({"model": model, "horizon": horizon, "seed": seed,
                                   **coordinate_scores(target, prediction, reference)})

            pca = fit_pca(train["x"], extra["pca_variance"], args.device)
            save_json(directory / "pca.json", {"rank": pca["rank"], "retained_variance": pca["retained_variance"],
                                               "fit": "training states only, raw centered coordinates"})
            dmd = fit_dmd(pca, train["x"], train["y1"])
            dmd_predictions = dmd_rollout(pca, dmd, val["x"], cfg["horizons"])
            np.savez_compressed(directory / "dmd.npz", mean=pca["mean"].cpu().numpy(),
                                vectors=pca["vectors"].cpu().numpy(), coefficient=dmd.cpu().numpy())
            for k in cfg["horizons"]:
                record("persistence", k, val["x"])
                record("train_mean", k, np.broadcast_to(train[f"y{k}"].mean(0), val[f"y{k}"].shape))
                record("dmd_recursive", k, dmd_predictions[k])
            for order in extra["var_orders"]:
                fit, alpha, table = fit_var(pca, train, val, order, base["ridge"]["alpha_grid"], args.device)
                selections[f"var_p{order}"] = table
                save_ridge(directory / f"var_p{order}.npz", fit)
                for k, prediction in var_rollout(pca, fit, val["lags"], order, cfg["horizons"]).items():
                    record(f"var_p{order}", k, prediction)
                del fit
            inputs = {
                "state": (train["x"], val["x"]),
                "history": (train["lags"][:, -2:].reshape(len(train["x"]), -1),
                            val["lags"][:, -2:].reshape(len(val["x"]), -1)),
                "observed": (np.concatenate([train["x"], train["raw_patch"]], axis=1),
                             np.concatenate([val["x"], val["raw_patch"]], axis=1))
            }
            for formulation, (tx, vx) in inputs.items():
                prepared = prepare_ridge(tx, args.device)
                for k in extra["mlp_horizons"]:
                    fit, alpha, table = select_ridge(prepared, train[f"y{k}"], vx, val[f"y{k}"],
                                                    base["ridge"]["alpha_grid"], val["ids"])
                    method = "ridge_direct" if formulation == "state" else "ridge_" + formulation
                    record(method, k, predict_ridge(fit, vx))
                    selections[method + f"_k{k}"] = table
                    save_ridge(directory / f"{method}_k{k}.npz", fit)
                    if formulation == "state" and k == 1:
                        for horizon, prediction in rollout_ridge(fit, vx, cfg["horizons"]).items():
                            record("ridge_recursive", horizon, prediction)
                    del fit
                if formulation == "state":
                    depth, alpha, table = select_ridge(prepared, train["depth"], vx, val["depth"],
                                                       base["ridge"]["alpha_grid"], val["ids"])
                    record("ridge_depth", 1, predict_ridge(depth, vx))
                    selections["ridge_depth"] = table
                    save_ridge(directory / "ridge_depth.npz", depth)
                    del depth
                if dataset["family"] == "control":
                    continue
                for k in extra["mlp_horizons"]:
                    mlp_cfg = matched_mlp_config(cfg["mlp"], tx.shape[1], train["x"].shape[1])
                    for seed in extra["seeds"]:
                        fit = fit_mlp(tx, train[f"y{k}"], vx, val[f"y{k}"], mlp_cfg, seed, args.device, val["ids"])
                        method = "mlp_" + formulation
                        record(method, k, predict_mlp(fit, vx), seed)
                        torch.save({"state_dict": fit["model"].state_dict(), "config": mlp_cfg,
                                    **{key: fit[key] for key in ("x_mean", "x_scale", "y_mean", "y_scale")}},
                                   directory / f"{method}_k{k}_seed{seed}.pt")
                        save_json(directory / f"{method}_k{k}_seed{seed}.json", {key: fit[key] for key in
                                  ("best_epoch", "history", "seconds", "parameters", "seed")})
                        print(name, protocol, method, k, seed, "epoch", fit["best_epoch"], flush=True)
                        del fit
                if formulation == "state":
                    for seed in extra["seeds"]:
                        fit = fit_mlp(tx, train["depth"], vx, val["depth"], cfg["mlp"], seed, args.device, val["ids"])
                        record("mlp_depth", 1, predict_mlp(fit, vx), seed)
                        torch.save({"state_dict": fit["model"].state_dict(), "config": cfg["mlp"],
                                    **{key: fit[key] for key in ("x_mean", "x_scale", "y_mean", "y_scale")}},
                                   directory / f"mlp_depth_seed{seed}.pt")
                        save_json(directory / f"mlp_depth_seed{seed}.json", {key: fit[key] for key in
                                  ("best_epoch", "history", "seconds", "parameters", "seed")})
                        del fit
                del prepared
            save_json(directory / "metrics.json", rows)
            write_rows(directory / "metrics.csv", rows)
            save_json(directory / "coordinate_r2.json", dimensions)
            save_json(directory / "selection.json", selections)
            save_json(directory / "complete.json", {"dataset": name, "protocol": protocol,
                      "anchor_sha256": anchor_hash, "seconds": time.perf_counter() - start, "test_evaluated": False})
            print("DONE", name, protocol, "seconds", round(time.perf_counter() - start), flush=True)
            del group, train, val, inputs
            torch.cuda.empty_cache()
    all_rows = []
    for directory in output.iterdir():
        if directory.is_dir() and (directory / "complete.json").exists():
            all_rows.extend(json.loads((directory / "metrics.json").read_text()))
    if all_rows:
        summaries = aggregate_results(all_rows, cfg["bootstrap_samples"], cfg["seed"])
        save_json(output / "summary.json", summaries)
        write_rows(output / "summary.csv", summaries)
        save_json(output / "depth_comparison.json", depth_comparison(all_rows, cfg["bootstrap_samples"], cfg["seed"]))


if __name__ == "__main__":
    main()
