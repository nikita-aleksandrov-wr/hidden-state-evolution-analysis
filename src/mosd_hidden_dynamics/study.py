import csv
import hashlib
import json
import time
from collections import defaultdict

import numpy as np

from .study_data import study_pairs
from .study_models import (diagnostic_metrics, fit_mlp, fit_ridge, predict_mlp, predict_ridge,
                           prepare_ridge, rollout_mlp, rollout_ridge, save_ridge, select_ridge)
from .utils import save_json, sha256_file, source_record


def write_rows(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.3f}" if isinstance(value, float) else value for key, value in row.items()})


def load_group(cfg, registry, run_dir, dataset, condition, protocol):
    chunks = {split: defaultdict(list) for split in ("train", "val")}
    manifest = []
    for record in registry:
        if record["dataset"] != dataset or record["role"] == "test":
            continue
        path = run_dir / "features" / condition / (record["series_id"] + ".npz")
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata["sha256"] != sha256_file(path) or metadata["data_sha256"] != record["data_sha256"]:
            raise ValueError("Feature/data checksum mismatch")
        if metadata["test_states_extracted"] or metadata["future_values_supplied"]:
            raise ValueError("Features violate the past-only policy")
        with np.load(path) as arrays:
            states, ends = arrays["states"].astype(np.float64), arrays["raw_endpoints"]
        if not np.all(ends < record["test_start"]):
            raise ValueError("Test states were extracted")
        data_path = run_dir / "data" / (record["series_id"] + ".npz")
        if sha256_file(data_path) != record["data_sha256"]:
            raise ValueError("Raw data checksum changed")
        with np.load(data_path) as data:
            observed = data["observed"][:record["test_start"]]
        windows = np.lib.stride_tricks.sliding_window_view(observed, cfg["context_length"])
        for split, indices in study_pairs(ends, record, cfg["context_length"], max(cfg["horizons"]), protocol).items():
            chunks[split]["x"].append(states[indices, 0])
            chunks[split]["history"].append(np.concatenate([states[indices - 1, 0], states[indices, 0]], axis=1))
            chunks[split]["depth"].append(states[indices, 1])
            chunks[split]["ids"].append(np.repeat(record["series_id"], len(indices)))
            chunks[split]["raw"].append(windows[ends[indices] - cfg["context_length"] + 1])
            for k in cfg["horizons"]:
                chunks[split][f"y{k}"].append(states[indices + k, 0])
            for index in indices:
                manifest.append({"series_id": record["series_id"], "split": split,
                                 "source_end": int(ends[index]), "receptive_start": int(ends[index] - 16 - cfg["context_length"] + 1),
                                 "max_target_end": int(ends[index] + max(cfg["horizons"]) * 16)})
    result = {split: {key: np.concatenate(values) for key, values in chunks[split].items()} for split in chunks}
    if protocol == "out_of_series" and set(result["train"]["ids"]) & set(result["val"]["ids"]):
        raise ValueError("Out-of-series memberships overlap")
    for series in {r["series_id"] for r in manifest}:
        train = [r for r in manifest if r["series_id"] == series and r["split"] == "train"]
        val = [r for r in manifest if r["series_id"] == series and r["split"] == "val"]
        if train and val:
            gap = min(r["receptive_start"] for r in val) - max(r["max_target_end"] for r in train) - 1
            if gap < cfg["context_length"]:
                raise ValueError("Common-anchor embargo failed")
    return result, manifest


def nearest_window_audit(train, val, threshold, device):
    import torch

    training_hashes = {hashlib.sha256(row.tobytes()).digest() for row in train}
    exact = sum(hashlib.sha256(row.tobytes()).digest() in training_hashes for row in val)
    def unit(x):
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        x = x - x.mean(dim=1, keepdim=True)
        return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    train = unit(train)
    maxima = []
    for offset in range(0, len(val), 256):
        block = unit(val[offset:offset + 256])
        best = torch.full((len(block),), -1., device=device)
        for start in range(0, len(train), 4096):
            best = torch.maximum(best, (block @ train[start:start + 4096].T).max(dim=1).values)
        maxima.extend(best.cpu().numpy().tolist())
    maxima = np.asarray(maxima)
    return {"exact_duplicate_validation_windows": int(exact), "threshold": threshold,
            "fraction_above_threshold": float(np.mean(maxima >= threshold)),
            "median_nearest_cosine": float(np.median(maxima)), "max_nearest_cosine": float(maxima.max()),
            "normalization": "each raw window centered and unit length; diagnostic only"}, maxima


def descriptive_statistics(group, season_raw):
    rows = []
    x = group["train"]["x"]
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1)
    raw = group["train"]["raw"][:, -16:]
    raw_mean, raw_scale = raw.mean(axis=0), raw.std(axis=0)
    raw_scale = np.where(raw_scale > 1e-8, raw_scale, 1)
    for split, data in group.items():
        for series in np.unique(data["ids"]):
            selected = data["ids"] == series
            for name, value in (("hidden", (data["x"][selected] - mean) / scale),
                                ("observed_patch", (data["raw"][selected, -16:] - raw_mean) / raw_scale)):
                change = np.diff(value, axis=0)
                energy = np.square(change).sum(axis=0)
                ordered = np.sort(energy)
                gini = float((2 * np.arange(1, len(ordered) + 1) - len(ordered) - 1) @ ordered / (len(ordered) * ordered.sum()))
                result = {"series_id": series, "split": split, "representation": name,
                          "relative_jump_rms": float(np.sqrt(np.square(change).mean() / np.square(value).mean())), "jump_energy_gini": gini}
                for lag in sorted({1, 4, max(1, season_raw // 16) if season_raw else 8}):
                    a, b = value[:-lag], value[lag:]
                    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
                    denominator = np.sqrt(np.square(a).sum(axis=0) * np.square(b).sum(axis=0))
                    valid = denominator > 1e-12
                    result[f"acf_lag_{lag}"] = float(np.median((a * b).sum(axis=0)[valid] / denominator[valid]))
                rows.append(result)
    return rows


def evaluate_group(cfg, base_cfg, group, output, condition, dataset, protocol, device, logger, skip_mlp):
    import torch

    train, val = group["train"], group["val"]
    rows, null_rows, selections = [], [], {}
    common = {"dataset": dataset["name"], "family": dataset["family"], "protocol": protocol, "condition": condition}
    def record(name, k, prediction, alpha=None, seed=None):
        target, reference = val[f"y{k}"], train[f"y{k}"].mean(axis=0)
        for series in np.unique(val["ids"]):
            mask = val["ids"] == series
            rows.append({**common, "model": name, "horizon": k, "seed": seed, "alpha": alpha,
                         "series_id": series, "n_train": len(train["x"]), "n_val": int(mask.sum()),
                         **diagnostic_metrics(target[mask], prediction[mask], reference)})
    start = time.perf_counter()
    prepared = prepare_ridge(train["x"], device)
    prepared_history = prepare_ridge(train["history"], device) if condition == "pretrained" else None
    ridge_rollouts, selected_one, one_step_alpha = None, None, None
    for k in cfg["horizons"]:
        y = train[f"y{k}"]
        model, alpha, table = select_ridge(prepared, y, val["x"], val[f"y{k}"], base_cfg["ridge"]["alpha_grid"], val["ids"])
        selections[f"ridge_direct_k{k}"] = table
        direct = predict_ridge(model, val["x"])
        if k == 1:
            ridge_rollouts = rollout_ridge(model, val["x"], cfg["horizons"])
            selected_one, one_step_alpha = model, alpha
            if condition == "pretrained":
                operator = model["coef"] / model["scale"][None, :]
                radius = float(torch.linalg.eigvals(operator).abs().max())
                save_json(output / "linear_diagnostics.json", {"spectral_radius": radius, "alpha": alpha,
                          "interpretation": "radius below one implies asymptotic convergence to a fixed point"})
        record("train_mean", k, np.broadcast_to(y.mean(axis=0), val[f"y{k}"].shape))
        record("persistence", k, val["x"])
        record("ridge_direct", k, direct, alpha)
        record("ridge_recursive", k, ridge_rollouts[k], one_step_alpha)
        if condition == "pretrained":
            save_ridge(output / f"ridge_k{k}.npz", model)
            history, history_alpha, history_table = select_ridge(prepared_history, y, val["history"], val[f"y{k}"],
                                                               base_cfg["ridge"]["alpha_grid"], val["ids"])
            selections[f"ridge_history_k{k}"] = history_table
            record("ridge_history", k, predict_ridge(history, val["history"]), history_alpha)
        logger.info("%s / %s / %s k=%d: ridge R2=%.3f", condition, dataset["name"], protocol, k,
                    np.mean([r["r2"] for r in rows if r["model"] == "ridge_direct" and r["horizon"] == k]))
    if condition == "pretrained":
        depth, alpha, table = select_ridge(prepared, train["depth"], val["x"], val["depth"],
                                          base_cfg["ridge"]["alpha_grid"], val["ids"])
        selections["ridge_depth_own_target_selection"] = table
        record("ridge_depth", 1, predict_ridge(depth, val["x"]), alpha)
        save_ridge(output / "ridge_depth.npz", depth)
        for kind in ("shuffle", "circular_shift"):
            for index in range(cfg["shuffled_seeds"]):
                rng = np.random.default_rng(cfg["seed"] + index)
                y = train["y1"].copy()
                if kind == "shuffle":
                    y = y[rng.permutation(len(y))]
                else:
                    for series in np.unique(train["ids"]):
                        mask = train["ids"] == series
                        shift = int(rng.integers(max(cfg["horizons"]) + 1, int(mask.sum()) - max(cfg["horizons"])))
                        y[mask] = np.roll(y[mask], shift, axis=0)
                control = fit_ridge(prepared, y, one_step_alpha)
                prediction = predict_ridge(control, val["x"])
                for series in np.unique(val["ids"]):
                    mask = val["ids"] == series
                    null_rows.append({**common, "control": kind, "seed": cfg["seed"] + index, "series_id": series,
                                      **diagnostic_metrics(val["y1"][mask], prediction[mask], train["y1"].mean(axis=0))})
        if dataset["family"] != "control" and not skip_mlp:
            for k in cfg["mlp"]["horizons"]:
                for seed in cfg["mlp"]["seeds"]:
                    fit = fit_mlp(train["x"], train[f"y{k}"], val["x"], val[f"y{k}"], cfg["mlp"], seed, device, val["ids"])
                    prediction = predict_mlp(fit, val["x"])
                    record("mlp_direct", k, prediction, seed=seed)
                    torch.save({"state_dict": fit["model"].state_dict(), "x_mean": fit["x_mean"], "x_scale": fit["x_scale"],
                                "y_mean": fit["y_mean"], "y_scale": fit["y_scale"], "config": cfg["mlp"]}, output / f"mlp_k{k}_seed{seed}.pt")
                    save_json(output / f"mlp_k{k}_seed{seed}.json", {key: fit[key] for key in ("best_epoch", "history", "seconds", "parameters", "seed")})
                    if k == 1:
                        for horizon, prediction in rollout_mlp(fit, val["x"], cfg["horizons"]).items():
                            record("mlp_recursive", horizon, prediction, seed=seed)
                    logger.info("MLP %s / %s k=%d seed=%d epoch=%d R2=%.3f (%.2f s)", dataset["name"], protocol, k, seed,
                                fit["best_epoch"], np.mean([r["r2"] for r in rows if r["model"] == "mlp_direct" and r["horizon"] == k and r["seed"] == seed]), fit["seconds"])
    save_json(output / "alpha_selection.json", selections)
    save_json(output / "metrics.json", rows)
    save_json(output / "null_controls.json", null_rows)
    write_rows(output / "metrics.csv", rows)
    write_rows(output / "null_controls.csv", null_rows)
    save_json(output / "complete.json", {**common, "seconds": time.perf_counter() - start,
              "models": sorted({r["model"] for r in rows}), "test_evaluated": False})
    return rows


def aggregate_results(rows, samples, seed):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in ("dataset", "protocol", "condition", "model", "horizon"))
        grouped[key].append(row)
    summaries = []
    for key, values in grouped.items():
        per_series = defaultdict(list)
        for value in values:
            per_series[value["series_id"]].append(value)
        means = {series: {metric: float(np.mean([r[metric] for r in trials if r.get(metric) is not None]))
                          for metric in ("r2", "cosine", "centered_cosine") if any(r.get(metric) is not None for r in trials)}
                 for series, trials in per_series.items()}
        summary = dict(zip(("dataset", "protocol", "condition", "model", "horizon"), key))
        summary["n_series"] = len(means)
        seeds = {r["seed"] for r in values if r.get("seed") is not None}
        summary["n_seeds"] = len(seeds) if seeds else 1
        if len(seeds) > 1:
            for metric in ("r2", "cosine"):
                summary[f"{metric}_std_over_seeds"] = float(np.std([np.mean([r[metric] for r in values if r["seed"] == seed]) for seed in seeds]))
        for metric in ("r2", "cosine", "centered_cosine"):
            available = [m[metric] for m in means.values() if metric in m]
            if available:
                summary[metric] = float(np.mean(available))
        if key[3] != "persistence":
            baseline = grouped[(key[0], key[1], key[2], "persistence", key[4])]
            for metric in ("r2", "cosine"):
                reference = {r["series_id"]: r[metric] for r in baseline}
                delta = np.asarray([means[series][metric] - reference[series] for series in means])
                indices = np.random.default_rng(seed).integers(0, len(delta), size=(samples, len(delta)))
                low, high = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
                summary.update({f"delta_{metric}": float(delta.mean()), f"delta_{metric}_ci_low": float(low), f"delta_{metric}_ci_high": float(high)})
        summaries.append(summary)
    return summaries


def depth_comparison(rows, samples, seed):
    groups = defaultdict(dict)
    for row in rows:
        if row["model"] in ("ridge_direct", "ridge_depth") and row["horizon"] == 1:
            key = (row["dataset"], row["protocol"], row["condition"])
            groups[key][(row["series_id"], row["model"])] = row
    results = []
    for key, values in groups.items():
        series = sorted({series for series, model in values if model == "ridge_depth"})
        if not series:
            continue
        result = dict(zip(("dataset", "protocol", "condition"), key))
        result["n_series"] = len(series)
        for metric in ("r2", "cosine"):
            delta = np.asarray([values[(series_id, "ridge_direct")][metric] - values[(series_id, "ridge_depth")][metric] for series_id in series])
            index = np.random.default_rng(seed).integers(0, len(delta), size=(samples, len(delta)))
            low, high = np.quantile(delta[index].mean(axis=1), [0.025, 0.975])
            result.update({f"delta_{metric}": float(delta.mean()), f"delta_{metric}_ci_low": float(low), f"delta_{metric}_ci_high": float(high)})
        result["local_threshold_met"] = all(result[f"delta_{m}"] >= .05 and result[f"delta_{m}_ci_low"] > 0 for m in ("r2", "cosine"))
        results.append(result)
    return results


def evaluate_study(cfg, base_cfg, registry, run_dir, repo, device, logger, evaluation_id,
                   selected_conditions=None, selected_datasets=None, skip_mlp=False):
    output = run_dir / "evaluations" / evaluation_id
    output.mkdir(parents=True, exist_ok=True)
    source = source_record(repo)
    manifest = {"source": source, "selection": "macro mean validation R2 over series; MLP seeds all retained",
                "seed": cfg["seed"], "common_anchors": True, "skip_mlp": skip_mlp, "test_evaluated": False}
    metadata_path = output / "evaluation_metadata.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old["source"]["sha256"] != source["sha256"] or old["skip_mlp"] != skip_mlp:
            raise ValueError("Evaluation source/settings changed; use a new evaluation ID")
    save_json(metadata_path, manifest)
    conditions = selected_conditions or ["pretrained"] + [f"random_{s}" for s in cfg["random_init_seeds"]]
    for condition in conditions:
        for dataset in cfg["datasets"]:
            if selected_datasets and dataset["name"] not in selected_datasets:
                continue
            for protocol in ("in_series", "out_of_series"):
                directory = output / f"{condition}_{dataset['name']}_{protocol}"
                if (directory / "complete.json").exists():
                    continue
                directory.mkdir(exist_ok=True)
                logger.info("Evaluating %s / %s / %s", condition, dataset["name"], protocol)
                group, anchors = load_group(cfg, registry, run_dir, dataset["name"], condition, protocol)
                write_rows(directory / "anchors.csv", anchors)
                if condition == "pretrained":
                    audit, nearest = nearest_window_audit(group["train"]["raw"], group["val"]["raw"], cfg["near_duplicate_cosine"], device)
                    save_json(directory / "window_audit.json", audit)
                    np.savez_compressed(directory / "nearest_windows.npz", maximum_cosine=nearest)
                    write_rows(directory / "descriptive.csv", descriptive_statistics(group, dataset.get("season_raw", 128)))
                evaluate_group(cfg, base_cfg, group, directory, condition, dataset, protocol, device, logger, skip_mlp)
                del group
                import torch
                torch.cuda.empty_cache()
                all_rows = []
                for path in output.glob("*/metrics.json"):
                    if (path.parent / "complete.json").exists():
                        all_rows.extend(json.loads(path.read_text()))
                summaries = aggregate_results(all_rows, cfg["bootstrap_samples"], cfg["seed"])
                save_json(output / "summary.json", summaries)
                write_rows(output / "summary.csv", summaries)
                comparisons = depth_comparison(all_rows, cfg["bootstrap_samples"], cfg["seed"])
                save_json(output / "depth_comparison.json", comparisons)
                write_rows(output / "depth_comparison.csv", comparisons)
