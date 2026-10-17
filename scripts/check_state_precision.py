import argparse
import copy
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Check FP16 storage against fresh FP32 states without retuning")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda:7")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from chronos.chronos2.model import Chronos2Model
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mosd_hidden_dynamics.extraction import extract_vectors, load_backbone
    from mosd_hidden_dynamics.study import load_group, write_rows
    from mosd_hidden_dynamics.study_models import diagnostic_metrics, fit_ridge, load_ridge, predict_ridge, prepare_ridge
    from mosd_hidden_dynamics.utils import save_json, source_record
    torch.set_num_threads(4)
    root = Path(args.run_dir).resolve()
    cfg = json.loads((root / "config_requested.json").read_text())
    base = json.loads((root / "backbone_config.json").read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    output = root / "precision_check"
    output.mkdir(exist_ok=False)
    pipeline, _ = load_backbone(base, args.device)
    rows = []
    for condition, seed in (("pretrained", None), (f"random_{cfg['random_init_seeds'][0]}", cfg["random_init_seeds"][0])):
        if seed is None:
            model = pipeline.model
        else:
            torch.manual_seed(seed)
            model = Chronos2Model(copy.deepcopy(pipeline.model.config)).to(device=args.device, dtype=torch.float32).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        for dataset in cfg["datasets"]:
            record = next(r for r in registry if r["dataset"] == dataset["name"] and r["role"] == "val")
            path = root / "features" / condition / (record["series_id"] + ".npz")
            with np.load(path) as data:
                ends, stored = data["raw_endpoints"][:128], data["states"][:128, 0].astype(np.float64)
            with np.load(root / "data" / (record["series_id"] + ".npz")) as data:
                observed = data["observed"]
            fresh = extract_vectors(model, observed, ends, [5], cfg["context_length"], cfg["batch_size"])[:, 0].astype(np.float64)
            identical = np.array_equal(fresh.astype(np.float16), stored.astype(np.float16))
            group, _ = load_group(cfg, registry, root, dataset["name"], condition, "out_of_series")
            train = group["train"]
            source = root / "evaluations/main" / f"{condition}_{dataset['name']}_out_of_series"
            if seed is None:
                fit = load_ridge(source / "ridge_k1.npz", args.device)
            else:
                table = json.loads((source / "alpha_selection.json").read_text())["ridge_direct_k1"]
                alpha = max(table, key=lambda row: (row["r2"], -row["alpha"]))["alpha"]
                fit = fit_ridge(prepare_ridge(train["x"], args.device), train["y1"], alpha)
            reference = train["y1"].mean(axis=0)
            first = diagnostic_metrics(stored[2:66], predict_ridge(fit, stored[1:65]), reference)
            second = diagnostic_metrics(fresh[2:66], predict_ridge(fit, fresh[1:65]), reference)
            rows.append({"dataset": dataset["name"], "condition": condition, "series_id": record["series_id"],
                         "fp16_roundtrip_identical": identical,
                         "relative_storage_error": float(np.linalg.norm(fresh - stored) / np.linalg.norm(fresh)),
                         "r2_fp16": first["r2"], "r2_fp32": second["r2"], "absolute_r2_change": abs(first["r2"] - second["r2"]),
                         "absolute_cosine_change": abs(first["cosine"] - second["cosine"])})
            print(dataset["name"], condition, f"R2 change={rows[-1]['absolute_r2_change']:.6f}, roundtrip={identical}", flush=True)
            del group
    save_json(output / "metrics.json", rows)
    write_rows(output / "metrics.csv", rows)
    save_json(output / "run_metadata.json", {"source": source_record(repo), "sample": "first 64 eligible one-step pairs of first unseen validation series",
              "hyperparameters_retuned": False, "pretrained_readout_refitted": False,
              "random_fit": "reconstructed from the same cached FP16 train inputs, targets and previously selected alpha", "test_evaluated": False})
    save_json(output / "complete.json", {"cases": len(rows), "max_absolute_r2_change": max(r["absolute_r2_change"] for r in rows),
              "max_absolute_cosine_change": max(r["absolute_cosine_change"] for r in rows),
              "all_fp16_roundtrips_identical": all(r["fp16_roundtrip_identical"] for r in rows), "test_evaluated": False})


if __name__ == "__main__":
    main()
