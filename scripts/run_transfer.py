import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Transfer frozen ridge maps between datasets without refitting")
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
    from mosd_hidden_dynamics.study import load_group, write_rows
    from mosd_hidden_dynamics.study_models import diagnostic_metrics, predict_ridge, rollout_ridge
    from mosd_hidden_dynamics.utils import save_json, sha256_file, source_record
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    root = Path(args.run_dir).resolve()
    cfg = json.loads((root / "config_requested.json").read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    evaluation = root / "evaluations" / args.evaluation_id
    output = evaluation / "transfer"
    output.mkdir(exist_ok=False)
    results = []
    hashes = {}
    for target in cfg["datasets"]:
        group, _ = load_group(cfg, registry, root, target["name"], "pretrained", "out_of_series")
        train, val = group["train"], group["val"]
        for source in cfg["datasets"]:
            directory = evaluation / f"pretrained_{source['name']}_out_of_series"
            recursive = None
            for horizon in cfg["horizons"]:
                path = directory / f"ridge_k{horizon}.npz"
                hashes[str(path.relative_to(root))] = sha256_file(path)
                with np.load(path) as data:
                    model = {name: torch.as_tensor(data[name], dtype=torch.float64, device=args.device) for name in data.files}
                if horizon == 1:
                    recursive = rollout_ridge(model, val["x"], cfg["horizons"])
                predictions = {"ridge_direct": predict_ridge(model, val["x"]), "ridge_recursive": recursive[horizon]}
                target_reference = train[f"y{horizon}"].mean(axis=0)
                for method, prediction in predictions.items():
                    for series in np.unique(val["ids"]):
                        mask = val["ids"] == series
                        results.append({"source_dataset": source["name"], "target_dataset": target["name"],
                                        "model": method, "horizon": horizon, "series_id": series,
                                        **diagnostic_metrics(val[f"y{horizon}"][mask], prediction[mask], target_reference)})
            print(f"Transferred {source['name']} -> {target['name']}", flush=True)
        save_json(output / "metrics.json", results)
    save_json(output / "run_metadata.json", {"source": source_record(repo), "fit_files": hashes,
              "normalization": "frozen source-training mean and scale; no target adaptation",
              "r2_reference": "target-training mean, used only as the evaluation baseline", "test_evaluated": False})
    write_rows(output / "metrics.csv", results)
    # A map transferred to its own dataset must reproduce its original held-out score.
    for dataset in cfg["datasets"]:
        original = json.loads((evaluation / f"pretrained_{dataset['name']}_out_of_series/metrics.json").read_text())
        for row in original:
            if row["model"] not in ("ridge_direct", "ridge_recursive"):
                continue
            match = next(r for r in results if r["source_dataset"] == r["target_dataset"] == dataset["name"]
                         and r["model"] == row["model"] and r["horizon"] == row["horizon"] and r["series_id"] == row["series_id"])
            np.testing.assert_allclose([match["r2"], match["cosine"]], [row["r2"], row["cosine"]], atol=1e-9, rtol=1e-9)
    save_json(output / "complete.json", {"source_datasets": len(cfg["datasets"]), "target_datasets": len(cfg["datasets"]),
              "horizons": cfg["horizons"], "self_transfer_verified": True, "test_evaluated": False})


if __name__ == "__main__":
    main()
