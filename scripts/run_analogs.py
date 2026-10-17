import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Raw-window analog forecasts and near-duplicate sensitivity")
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
    from mosd_hidden_dynamics.study_models import (diagnostic_metrics, load_mlp, load_ridge, nearest_indices,
                                                  predict_mlp, predict_ridge, rollout_ridge)
    from mosd_hidden_dynamics.utils import save_json, source_record
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root = Path(args.run_dir).resolve()
    cfg = json.loads((root / "config_requested.json").read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    evaluation = root / "evaluations" / args.evaluation_id
    output = evaluation / "analogs"
    output.mkdir(exist_ok=False)
    save_json(output / "run_metadata.json", {"source": source_record(repo), "threshold": cfg["near_duplicate_cosine"],
              "selection": "fixed k=1/5; training neighbors only; no alpha/epoch retuning on filtered subsets", "test_evaluated": False})
    rows, coverage = [], []
    for dataset in cfg["datasets"]:
        for protocol in ("in_series", "out_of_series"):
            group, _ = load_group(cfg, registry, root, dataset["name"], "pretrained", protocol)
            train, val = group["train"], group["val"]
            index, similarity = nearest_indices(train["raw"], val["raw"], 5, args.device)
            far = similarity[:, 0] < cfg["near_duplicate_cosine"]
            for series in np.unique(val["ids"]):
                mask = val["ids"] == series
                coverage.append({"dataset": dataset["name"], "protocol": protocol, "series_id": series,
                                 "total_pairs": int(mask.sum()), "retained_pairs": int((mask & far).sum())})
            source = evaluation / f"pretrained_{dataset['name']}_{protocol}"
            recursive = rollout_ridge(load_ridge(source / "ridge_k1.npz", args.device), val["x"], cfg["horizons"])
            for k in cfg["horizons"]:
                predictions = [("raw_knn1", None, train[f"y{k}"][index[:, 0]]),
                               ("raw_knn5", None, train[f"y{k}"][index].mean(axis=1)),
                               ("persistence", None, val["x"]),
                               ("train_mean", None, np.broadcast_to(train[f"y{k}"].mean(axis=0), val[f"y{k}"].shape)),
                               ("ridge_direct", None, predict_ridge(load_ridge(source / f"ridge_k{k}.npz", args.device), val["x"])),
                               ("ridge_recursive", None, recursive[k])]
                if dataset["family"] != "control" and k in cfg["mlp"]["horizons"]:
                    for seed in cfg["mlp"]["seeds"]:
                        model = load_mlp(source / f"mlp_k{k}_seed{seed}.pt", args.device)
                        predictions.append(("mlp_direct", seed, predict_mlp(model, val["x"])))
                for name, seed, prediction in predictions:
                    for subset, retained in (("all", np.ones(len(far), dtype=bool)), ("far_windows", far)):
                        for series in np.unique(val["ids"]):
                            mask = (val["ids"] == series) & retained
                            if mask.sum() < 2:
                                continue
                            rows.append({"dataset": dataset["name"], "protocol": protocol, "condition": "pretrained",
                                         "model": name, "horizon": k, "seed": seed, "subset": subset, "series_id": series,
                                         "n_val": int(mask.sum()), **diagnostic_metrics(val[f"y{k}"][mask], prediction[mask], train[f"y{k}"].mean(axis=0))})
            save_json(output / "metrics.json", rows)
            write_rows(output / "coverage.csv", coverage)
            print(f"{dataset['name']} / {protocol}: retained {far.mean():.3f} after raw-window filter", flush=True)
            del group
            torch.cuda.empty_cache()
    summaries = []
    for subset in ("all", "far_windows"):
        summaries.extend({"subset": subset, **r} for r in aggregate_results([r for r in rows if r["subset"] == subset], cfg["bootstrap_samples"], cfg["seed"]))
    save_json(output / "summary.json", summaries)
    write_rows(output / "summary.csv", summaries)
    write_rows(output / "metrics.csv", rows)
    save_json(output / "complete.json", {"test_evaluated": False, "groups": len(coverage)})


if __name__ == "__main__":
    main()
