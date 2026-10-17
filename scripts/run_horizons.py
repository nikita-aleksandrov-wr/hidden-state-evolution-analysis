import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Direct and recursive forecasts from a recorded checkpoint")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:7")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mosd_hidden_dynamics.splits import audit_manifest, make_manifest, save_manifest
    from mosd_hidden_dynamics.study_models import diagnostic_metrics, predict_ridge, prepare_ridge, rollout_ridge, select_ridge
    from mosd_hidden_dynamics.utils import save_json, source_record

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    source, output = Path(args.source_run), Path(args.output)
    assert not output.resolve().is_relative_to(repo)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    cfg = json.loads((source / "config_requested.json").read_text())
    horizons = [1, 4, 8, 16, 32, 64]
    with np.load(source / "data.npz") as data:
        observed = data["observed"]
    with np.load(source / "states.npz") as data:
        states, ends = data["states"][:, 0].astype(np.float64), data["raw_endpoints"]
    lookup = {int(end): i for i, end in enumerate(ends)}
    rows, bounds = make_manifest(len(observed), cfg["state_extraction"]["context_length_raw"], 16, max(horizons), **cfg["split"])
    audit = audit_manifest(rows, bounds, observed)
    save_manifest(output / "split_manifest.csv", rows)
    selected = {s: [r for r in rows if r["eligible"] and r["split"] == s] for s in ("train", "val")}
    x = {s: states[[lookup[r["raw_context_end"]] for r in selected[s]]] for s in selected}
    y = {s: {k: states[[lookup[r["raw_context_end"] + k * 16] for r in selected[s]]] for k in horizons} for s in selected}
    prepared = prepare_ridge(x["train"], args.device)
    results, tables = [], {}
    rollouts = None
    for k in horizons:
        model, alpha, table = select_ridge(prepared, y["train"][k], x["val"], y["val"][k], cfg["ridge"]["alpha_grid"])
        tables[str(k)] = table
        if k == 1:
            rollouts = rollout_ridge(model, x["val"], horizons)
        predictions = {"train_mean": np.broadcast_to(y["train"][k].mean(axis=0), y["val"][k].shape),
                       "persistence": x["val"], "ridge_direct": predict_ridge(model, x["val"]),
                       "ridge_recursive": rollouts[k]}
        for name, prediction in predictions.items():
            scores = diagnostic_metrics(y["val"][k], prediction, y["train"][k].mean(axis=0))
            results.append({"horizon": k, "model": name, "alpha": alpha if name == "ridge_direct" else None, **scores})
            print(f"k={k:2d} {name:16s} R2={scores['r2']:.3f} cosine={scores['cosine']:.3f}", flush=True)
    save_json(output / "metrics.json", results)
    save_json(output / "alpha_selection.json", tables)
    save_json(output / "run_metadata.json", {"source_run": str(source), "source": source_record(repo),
              "horizons": horizons, "common_anchors": True, "split_audit": audit,
              "test_evaluated": False, "seconds": time.perf_counter() - start, "status": "complete"})
    with (output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: f"{value:.3f}" if isinstance(value, float) else value for key, value in row.items()} for row in results)


if __name__ == "__main__":
    main()
