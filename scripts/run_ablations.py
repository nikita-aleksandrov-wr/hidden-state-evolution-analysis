import argparse
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Context, layer, scale, position and observation controls")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="configs/remaining_hidden_dynamics.json")
    parser.add_argument("--device", required=True)
    parser.add_argument("--stage", choices=("extract", "evaluate"), required=True)
    parser.add_argument("--variants", nargs="+")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mosd_hidden_dynamics.ablations import extract_ablations, load_ablation_group, variant_specs
    from mosd_hidden_dynamics.extended import series_metrics
    from mosd_hidden_dynamics.study import aggregate_results, descriptive_statistics, write_rows
    from mosd_hidden_dynamics.study_models import predict_ridge, prepare_ridge, save_ridge, select_ridge
    from mosd_hidden_dynamics.utils import save_json, source_record
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(args.run_dir).resolve()
    cfg = json.loads((root / "config_requested.json").read_text())
    base = json.loads((root / "backbone_config.json").read_text())
    extra = json.loads(Path(args.config).read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    output = root / "ablations"
    output.mkdir(exist_ok=True)
    requested = output / "config_requested.json"
    if requested.exists() and json.loads(requested.read_text()) != extra:
        raise ValueError("Ablation settings changed")
    save_json(requested, extra)
    if args.stage == "extract":
        extract_ablations(cfg, base, extra, registry, root, args.device, args.variants)
        return
    source = source_record(repo)
    cases = []
    for spec in variant_specs(extra):
        labels = ["block0", "block5", "block11", "final_norm"] if spec["name"].startswith("context_") or spec.get("shuffle_seed") else ["block5"]
        cases.extend({"variant": spec["name"], "label": label, "stride": 16, "calibrate": False,
                      "horizons": [1, 4], "common_context": 1024} for label in labels)
    cases.extend({"variant": "context_512", "label": "block5", "stride": stride, "calibrate": False,
                  "horizons": [1], "common_context": 512} for stride in (64, 512))
    cases.append({"variant": "context_512", "label": "block5", "stride": 16, "calibrate": True,
                  "horizons": [1, 4], "common_context": 1024})
    for case in cases:
        if args.variants and case["variant"] not in args.variants:
            continue
        condition = f"{case['variant']}_{case['label']}_stride{case['stride']}_calibration{case['calibrate']}"
        for dataset in cfg["datasets"]:
            name = dataset["name"]
            if case["variant"].startswith("noise_") and dataset["family"] != "chaotic":
                continue
            if case["variant"] == "matched_ar1" and dataset["family"] == "control":
                continue
            for protocol in ("in_series", "out_of_series"):
                directory = output / "evaluations" / f"{condition}_{name}_{protocol}"
                if (directory / "complete.json").exists():
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                start = time.perf_counter()
                group, manifest, digest = load_ablation_group(cfg, registry, root, name, protocol,
                    case["variant"], case["label"], case["horizons"], case["stride"], case["calibrate"],
                    extra["ablations"]["calibration_states"], case["common_context"])
                train, val = group["train"], group["val"]
                common = {"dataset": name, "family": dataset["family"], "protocol": protocol, "condition": condition}
                rows, selections = [], {}
                write_rows(directory / "anchors.csv", manifest)
                prepared = prepare_ridge(train["x"], args.device)
                raw_prepared = None
                if case["variant"] == "context_512" and case["label"] == "block5" and case["stride"] == 16 and not case["calibrate"]:
                    x = torch.as_tensor(train["x"], device=args.device, dtype=torch.float64)
                    mean = x.mean(0)
                    centered = x - mean
                    values, vectors = torch.linalg.eigh(centered.T @ centered)
                    raw_prepared = {"x": centered, "mean": mean, "scale": torch.ones_like(mean),
                                    "z_mean": centered.mean(0), "values": values.clamp_min(0), "vectors": vectors}
                calibrated = prepare_ridge(train["z"], args.device) if case["calibrate"] else None
                for k in case["horizons"]:
                    reference = train[f"y{k}"].mean(0)
                    predictions = [("persistence", val["x"]),
                                   ("train_mean", np.broadcast_to(reference, val[f"y{k}"].shape))]
                    model, alpha, table = select_ridge(prepared, train[f"y{k}"], val["x"], val[f"y{k}"],
                                                       base["ridge"]["alpha_grid"], val["ids"])
                    predictions.append(("ridge_direct", predict_ridge(model, val["x"])))
                    selections[f"ridge_k{k}"] = table
                    save_ridge(directory / f"ridge_k{k}.npz", model)
                    if raw_prepared is not None:
                        fit, alpha, table = select_ridge(raw_prepared, train[f"y{k}"], val["x"], val[f"y{k}"],
                                                        base["ridge"]["alpha_grid"], val["ids"])
                        predictions.append(("ridge_unscaled", predict_ridge(fit, val["x"])))
                        selections[f"ridge_unscaled_k{k}"] = table
                    if calibrated is not None:
                        fit, alpha, table = select_ridge(calibrated, train[f"zy{k}"], val["z"], val[f"zy{k}"],
                                                        base["ridge"]["alpha_grid"], val["ids"])
                        prediction = predict_ridge(fit, val["z"]) * val["calibration_scale"] + val["calibration_mean"]
                        predictions.append(("ridge_prefix_standardized", prediction))
                        selections[f"ridge_prefix_standardized_k{k}"] = table
                    for method, prediction in predictions:
                        rows.extend({**common, "model": method, "horizon": k, "seed": None, **r}
                                    for r in series_metrics(val[f"y{k}"], prediction, reference, val["ids"]))
                descriptions = descriptive_statistics(group, dataset.get("season_raw", 128))
                for split, part in group.items():
                    for series in np.unique(part["ids"]):
                        mask = part["ids"] == series
                        for label, value in (("hidden_native", part["x"][mask]), ("observed_patch_native", part["raw"][mask, -16:])):
                            if len(value) < 2:
                                continue
                            a, b = value[:-1], value[1:]
                            norm = np.linalg.norm(a, axis=1).clip(1e-12)
                            cosine = np.sum(a * b, axis=1) / (norm * np.linalg.norm(b, axis=1).clip(1e-12))
                            descriptions.append({"series_id": str(series), "split": split, "representation": label,
                                "mean_relative_adjacent_jump": float(np.mean(np.linalg.norm(b - a, axis=1) / norm)),
                                "lag1_cosine": float(np.mean(cosine))})
                save_json(directory / "metrics.json", rows)
                write_rows(directory / "metrics.csv", rows)
                save_json(directory / "descriptive.json", descriptions)
                save_json(directory / "selection.json", selections)
                save_json(directory / "run_metadata.json", {"source": source, "case": case, "anchor_sha256": digest,
                          "calibration": "first 128 past states per entity; prefix excluded from fitting/evaluation anchors" if case["calibrate"] else None,
                          "stride_warning": "Changing stride changes both overlap and physical prediction horizon.",
                          "test_evaluated": False})
                save_json(directory / "complete.json", {"seconds": time.perf_counter() - start, "test_evaluated": False})
                print("DONE", condition, name, protocol, flush=True)
                del group, train, val, prepared
                torch.cuda.empty_cache()
    rows = []
    for path in (output / "evaluations").glob("*/metrics.json"):
        if (path.parent / "complete.json").exists():
            rows.extend(json.loads(path.read_text()))
    if rows:
        summaries = aggregate_results(rows, cfg["bootstrap_samples"], cfg["seed"])
        save_json(output / "summary.json", summaries)
        write_rows(output / "summary.csv", summaries)


if __name__ == "__main__":
    main()
