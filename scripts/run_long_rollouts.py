import argparse
import copy
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Frozen models rolled forward as far as 256 analysis steps")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--evaluation-id", default="main")
    parser.add_argument("--device", default="cuda:7")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mosd_hidden_dynamics.study import aggregate_results, load_group, write_rows
    from mosd_hidden_dynamics.study_models import diagnostic_metrics, load_mlp, load_ridge, predict_mlp, rollout_ridge
    from mosd_hidden_dynamics.utils import save_json, sha256_file, source_record
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root = Path(args.run_dir).resolve()
    cfg = json.loads((root / "config_requested.json").read_text())
    cfg = copy.deepcopy(cfg)
    cfg["horizons"] = [1, 4, 8, 16, 32, 64, 128, 256]
    registry = json.loads((root / "series_registry.json").read_text())
    evaluation = root / "evaluations" / args.evaluation_id
    output = evaluation / "long_rollouts"
    output.mkdir(exist_ok=False)
    rows, failures, hashes = [], [], {}
    for dataset in cfg["datasets"]:
        if dataset["family"] == "control":
            continue
        group, _ = load_group(cfg, registry, root, dataset["name"], "pretrained", "out_of_series")
        train, val = group["train"], group["val"]
        source = evaluation / f"pretrained_{dataset['name']}_out_of_series"
        predictions = [("train_mean", None, {k: np.broadcast_to(train[f"y{k}"].mean(axis=0), val[f"y{k}"].shape) for k in cfg["horizons"]}),
                       ("persistence", None, {k: val["x"] for k in cfg["horizons"]})]
        path = source / "ridge_k1.npz"
        hashes[str(path.relative_to(root))] = sha256_file(path)
        predictions.append(("ridge_recursive", None, rollout_ridge(load_ridge(path, args.device), val["x"], cfg["horizons"])))
        for seed in cfg["mlp"]["seeds"]:
            path = source / f"mlp_k1_seed{seed}.pt"
            hashes[str(path.relative_to(root))] = sha256_file(path)
            fit = load_mlp(path, args.device)
            state, predicted = val["x"].copy(), {}
            failed_step = None
            for step in range(1, max(cfg["horizons"]) + 1):
                state = predict_mlp(fit, state)
                if not np.isfinite(state).all():
                    failed_step = step
                    break
                if step in cfg["horizons"]:
                    predicted[step] = state.copy()
            if failed_step is not None:
                failures.append({"dataset": dataset["name"], "seed": seed, "first_non_finite_step": failed_step})
            predictions.append(("mlp_recursive", seed, predicted))
        for method, seed, predicted in predictions:
            for k in cfg["horizons"]:
                for series in np.unique(val["ids"]):
                    mask = val["ids"] == series
                    row = {"dataset": dataset["name"], "family": dataset["family"], "protocol": "out_of_series",
                           "condition": "pretrained", "model": method, "horizon": k, "seed": seed,
                           "series_id": series, "n_val": int(mask.sum())}
                    if k not in predicted:
                        row.update(status="non_finite_prediction", r2=None, cosine=None)
                    else:
                        row.update(status="finite", **diagnostic_metrics(val[f"y{k}"][mask], predicted[k][mask], train[f"y{k}"].mean(axis=0)))
                    rows.append(row)
        save_json(output / "metrics.json", rows)
        print(f"Completed frozen 256-step rollouts for {dataset['name']}", flush=True)
    write_rows(output / "metrics.csv", rows)
    complete_keys = {(r["dataset"], r["model"], r["horizon"]) for r in rows if r["status"] != "finite"}
    valid = [r for r in rows if (r["dataset"], r["model"], r["horizon"]) not in complete_keys]
    summaries = aggregate_results(valid, cfg["bootstrap_samples"], cfg["seed"])
    for dataset, method, k in complete_keys:
        summaries.append({"dataset": dataset, "model": method, "horizon": k, "status": "non_finite_seed; method not averaged"})
    save_json(output / "summary.json", summaries)
    write_rows(output / "summary.csv", summaries)
    save_json(output / "run_metadata.json", {"source": source_record(repo), "horizons": cfg["horizons"], "frozen_fit_files": hashes,
              "common_anchors": "eligible for k=256; source models and their scalers unchanged",
              "refitted": False, "r2_reference": "training targets at the same long-horizon anchors", "failures": failures,
              "test_evaluated": False})
    save_json(output / "complete.json", {"horizons": cfg["horizons"], "failed_fits": len(failures), "test_evaluated": False})


if __name__ == "__main__":
    main()
