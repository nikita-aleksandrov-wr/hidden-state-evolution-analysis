import argparse
import copy
import json
import os
from pathlib import Path
import sys

import numpy as np


def continued_context(window, continuation):
    window, continuation = np.asarray(window), np.asarray(continuation)
    if window.ndim != 1 or continuation.ndim != 1 or not len(continuation):
        raise ValueError("A past window and a nonempty predicted continuation are required")
    return np.concatenate([window, continuation])[-len(window):]


def cached_states(root, anchors, condition, horizon=0):
    result = np.empty((len(anchors), 768), dtype=np.float32)
    ids = np.asarray([r["series_id"] for r in anchors])
    for sid in np.unique(ids):
        indices = np.flatnonzero(ids == sid)
        with np.load(root / "features" / condition / (str(sid) + ".npz")) as data:
            ends = np.asarray([anchors[i]["source_end"] + 16 * horizon for i in indices])
            positions = np.searchsorted(data["raw_endpoints"], ends)
            np.testing.assert_array_equal(data["raw_endpoints"][positions], ends)
            result[indices] = data["states"][positions, 0]
    return result


def main():
    parser = argparse.ArgumentParser(description="Input forecasting followed by matched past-only state re-encoding")
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--stop-file", required=True)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    root = Path(args.reference_run).resolve()
    os.environ["HF_HOME"] = str(root.parents[1] / "cache/huggingface")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    import torch
    from chronos.chronos2.model import Chronos2Model
    from mosd_hidden_dynamics.consistent import fixed_starts, score_series, scoring_space
    from mosd_hidden_dynamics.extended import load_extended_group
    from mosd_hidden_dynamics.extraction import extract_vectors, load_backbone
    from mosd_hidden_dynamics.utils import save_json, source_record
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads((repo / "configs/consistent_hidden_dynamics.json").read_text())
    base = json.loads((root / "config_requested.json").read_text())
    backbone = json.loads((root / "backbone_config.json").read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    out = Path(args.output_dir).resolve() / "reencoding"
    out.mkdir(parents=True, exist_ok=True)
    assert not out.is_relative_to(repo)
    requested = out / "config_requested.json"
    if requested.exists():
        assert json.loads(requested.read_text()) == cfg
    save_json(requested, cfg)
    source = source_record(repo)
    pipeline, load_seconds = load_backbone(backbone, args.device)
    quantiles = np.asarray(pipeline.model.chronos_config.quantiles)
    median = int(np.argmin(abs(quantiles - .5)))
    assert abs(quantiles[median] - .5) < 1e-8
    def stopped():
        if Path(args.stop_file).exists():
            raise InterruptedError("Stop requested; completed dataset/encoder controls remain saved")
    def encode(model, windows):
        length = windows.shape[1]
        ends = np.arange(length - 1, windows.size, length)
        return extract_vectors(model, windows.reshape(-1), ends, [5], length, 32)[:, 0]
    for dataset in cfg["datasets"] + ["white_noise"]:
        stopped()
        directory = out / dataset
        directory.mkdir(exist_ok=True)
        long_cfg = copy.deepcopy(base)
        long_cfg["horizons"] = cfg["horizons"]
        group, all_anchors, digest = load_extended_group(long_cfg, registry, root, dataset, "out_of_series", order=8, segment=4)
        anchors = [r for r in all_anchors if r["split"] == "val"]
        train_long = [r for r in all_anchors if r["split"] == "train"]
        indices = fixed_starts(group["val"]["ids"], cfg["plugin_starts_per_series"])
        selected = [anchors[i] for i in indices]
        ids = group["val"]["ids"][indices]
        _, short_anchors, _ = load_extended_group(base, registry, root, dataset, "out_of_series", order=8, segment=4)
        train_short = [r for r in short_anchors if r["split"] == "train"]
        del group
        save_json(directory / "selected_anchors.json", selected)
        observations = {}
        for record in registry:
            if record["dataset"] == dataset and record["role"] != "test":
                with np.load(root / "data" / (record["series_id"] + ".npz")) as data:
                    observations[record["series_id"]] = data["observed"][:record["test_start"]]
        raw_train = np.concatenate([observations[r["series_id"]] for r in registry if r["dataset"] == dataset and r["role"] == "train"])
        raw_scale = max(float(raw_train.std()), 1e-8)
        windows = np.stack([observations[a["series_id"]][a["source_end"] - 511:a["source_end"] + 1] for a in selected])
        continuations, raw_scores = {}, []
        for horizon in cfg["plugin_horizons"]:
            stopped()
            cache = directory / f"input_forecast_k{horizon}.npz"
            if cache.exists():
                with np.load(cache) as data:
                    forecast = data["forecast"]
            else:
                predictions = pipeline.predict([w.copy() for w in windows], prediction_length=16 * horizon,
                    context_length=512, batch_size=16, cross_learning=False, limit_prediction_length=False)
                forecast = np.stack([p[0, median].cpu().numpy() for p in predictions])
                assert forecast.shape == (len(selected), 16 * horizon)
                np.savez_compressed(cache, forecast=forecast)
            true_future = np.stack([observations[a["series_id"]][a["source_end"] + 1:a["source_end"] + 1 + 16 * horizon] for a in selected])
            true_windows = np.stack([continued_context(w, f) for w, f in zip(windows, true_future)])
            continuations[horizon] = {"truth": true_windows}
            for method, future in (("chronos_median", forecast), ("last_value", np.repeat(windows[:, -1:], 16 * horizon, axis=1))):
                predicted_windows = np.stack([continued_context(w, f) for w, f in zip(windows, future)])
                continuations[horizon][method] = predicted_windows
                errors = np.sqrt(np.square(predicted_windows - true_windows).mean(axis=1)) / raw_scale
                for i, value in enumerate(errors):
                    raw_scores.append({"dataset": dataset, "horizon": horizon, "method": method,
                        "series_id": str(ids[i]), "selected_index": i, "raw_window_nrmse": float(value)})
            print("INPUT FORECAST", dataset, horizon, len(selected), flush=True)
        save_json(directory / "input_errors.json", raw_scores)
        for condition, seed in [("pretrained", None)] + [(f"random_{s}", s) for s in cfg["seeds"]]:
            stopped()
            complete = directory / f"{condition}_complete.json"
            if complete.exists():
                continue
            if seed is None:
                model = pipeline.model
            else:
                torch.manual_seed(seed)
                model = Chronos2Model(copy.deepcopy(pipeline.model.config)).to(device=args.device, dtype=torch.float32).eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
            train_x = cached_states(root, train_short, condition)
            space = scoring_space(train_x, args.device)
            initial = cached_states(root, selected, condition)
            rows, correlations, oracle_checks = [], [], []
            saved_predictions = {"selected_indices": indices, "ids": ids}
            for horizon in cfg["plugin_horizons"]:
                stopped()
                truth = cached_states(root, selected, condition, horizon)
                reference = cached_states(root, train_long, condition, horizon).mean(0)
                oracle = encode(model, continuations[horizon]["truth"])
                relative = np.linalg.norm(oracle - truth, axis=1) / np.linalg.norm(truth, axis=1).clip(1e-12)
                oracle_checks.append({"horizon": horizon, "median_relative_l2": float(np.median(relative)), "max_relative_l2": float(relative.max())})
                assert float(relative.max()) < .002, "Re-encoded true windows disagree with cached representation beyond FP16 tolerance"
                saved_predictions[f"target{horizon}"] = truth
                for method in ("chronos_median", "last_value", "persistence"):
                    predicted = initial if method == "persistence" else encode(model, continuations[horizon][method])
                    saved_predictions[f"{method}_k{horizon}"] = predicted
                    rows.extend({"dataset": dataset, "condition": condition, "seed": seed, "method": method, "horizon": horizon, **r}
                                for r in score_series(truth, predicted, reference, ids, space))
                    if method != "persistence":
                        latent = np.linalg.norm(predicted - truth, axis=1) / np.linalg.norm(truth - reference, axis=1).clip(1e-12)
                        for i, value in enumerate(latent):
                            raw = next(r["raw_window_nrmse"] for r in raw_scores if r["method"] == method and r["horizon"] == horizon and r["selected_index"] == i)
                            correlations.append({"horizon": horizon, "method": method, "series_id": str(ids[i]),
                                "selected_index": i, "raw_window_nrmse": raw, "latent_relative_error": float(value)})
                print("REENCODE", dataset, condition, horizon, flush=True)
            save_json(directory / f"{condition}_metrics.json", rows)
            save_json(directory / f"{condition}_errors.json", correlations)
            save_json(directory / f"{condition}_oracle_checks.json", oracle_checks)
            np.savez_compressed(directory / f"{condition}_predictions.npz", **saved_predictions)
            save_json(complete, {"status": "complete", "test_evaluated": False})
            if seed is not None:
                del model
                torch.cuda.empty_cache()
        save_json(directory / "run_metadata.json", {"source": source, "report_anchor_sha256": digest,
            "selected_indices": indices.tolist(), "test_evaluated": False, "context": 512,
            "forecast": "pretrained Chronos-2 median, one independent task per past-only window",
            "encoder_conditions": ["pretrained"] + [f"random_{s}" for s in cfg["seeds"]],
            "interpretation": "Within-target comparison; a readout win does not prove extra information or an autonomous law"})
    save_json(out / "complete.json", {"status": "complete", "model_load_seconds": load_seconds, "test_evaluated": False})


if __name__ == "__main__":
    main()
