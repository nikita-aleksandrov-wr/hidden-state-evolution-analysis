import hashlib
import json
from collections import defaultdict

import numpy as np

from .study_data import study_pairs
from .study_models import diagnostic_metrics, fit_ridge, predict_ridge, prepare_ridge
from .utils import sha256_file


def load_extended_group(cfg, registry, root, dataset, protocol, order=8, segment=4):
    chunks = {name: defaultdict(list) for name in ("train", "val")}
    anchors = []
    for record in registry:
        if record["dataset"] != dataset or record["role"] == "test":
            continue
        path = root / "features/pretrained" / (record["series_id"] + ".npz")
        metadata = json.loads(path.with_suffix(".json").read_text())
        if (sha256_file(path) != metadata["sha256"] or metadata["test_states_extracted"]
                or metadata["future_values_supplied"] or metadata["data_sha256"] != record["data_sha256"]):
            raise ValueError("Feature integrity/test policy failed")
        with np.load(path) as data:
            states, ends = data["states"].astype(np.float32), data["raw_endpoints"]
        if np.any(ends >= record["test_start"]):
            raise ValueError("Test states present")
        raw_path = root / "data" / (record["series_id"] + ".npz")
        if sha256_file(raw_path) != record["data_sha256"]:
            raise ValueError("Raw data changed")
        with np.load(raw_path) as data:
            observed = data["observed"][:record["test_start"]]
        for split, indices in study_pairs(ends, record, cfg["context_length"], max(cfg["horizons"]), protocol).items():
            boundary = int(record["length"] * .7) + cfg["context_length"] if protocol == "in_series" and split == "val" else 0
            indices = indices[ends[indices] - (order - 1) * 16 - cfg["context_length"] + 1 >= boundary]
            if not len(indices):
                raise ValueError("Empty common history anchors")
            part = chunks[split]
            part["x"].append(states[indices, 0])
            part["depth"].append(states[indices, 1])
            part["lags"].append(np.stack([states[indices - lag, 0] for lag in range(order - 1, -1, -1)], axis=1))
            part["trajectory"].append(np.stack([states[indices + k, 0] for k in range(segment + 1)], axis=1))
            part["raw_patch"].append(np.stack([observed[e - 15:e + 1] for e in ends[indices]]))
            part["ids"].append(np.repeat(record["series_id"], len(indices)))
            for k in cfg["horizons"]:
                part[f"y{k}"].append(states[indices + k, 0])
            anchors.extend({"series_id": record["series_id"], "split": split, "source_end": int(ends[i]),
                            "receptive_start": int(ends[i] - (order - 1) * 16 - cfg["context_length"] + 1),
                            "max_target_end": int(ends[i] + max(cfg["horizons"]) * 16)} for i in indices)
    group = {split: {key: np.concatenate(values) for key, values in part.items()} for split, part in chunks.items()}
    if protocol == "out_of_series" and set(group["train"]["ids"]) & set(group["val"]["ids"]):
        raise ValueError("Entity overlap")
    for series in {a["series_id"] for a in anchors}:
        a = [r for r in anchors if r["series_id"] == series and r["split"] == "train"]
        b = [r for r in anchors if r["series_id"] == series and r["split"] == "val"]
        if a and b and min(r["receptive_start"] for r in b) - max(r["max_target_end"] for r in a) - 1 < cfg["context_length"]:
            raise ValueError("Embargo violation")
    digest = hashlib.sha256(json.dumps(anchors, sort_keys=True).encode()).hexdigest()
    return group, anchors, digest


def fit_pca(x, variance, device):
    import torch
    tx = torch.as_tensor(x, dtype=torch.float64, device=device)
    mean = tx.mean(0)
    values, vectors = torch.linalg.eigh((tx - mean).T @ (tx - mean))
    values = values.flip(0).clamp_min(0)
    vectors = vectors.flip(1)
    if not 0 < variance <= 1 or float(values.sum()) <= 0:
        raise ValueError("PCA needs nonconstant training states and variance in (0, 1]")
    rank = min(len(values), int(torch.searchsorted(values.cumsum(0) / values.sum(), variance)) + 1)
    return {"mean": mean, "vectors": vectors[:, :rank], "rank": rank,
            "retained_variance": float(values[:rank].sum() / values.sum())}


def project(pca, x):
    import torch
    value = torch.as_tensor(x, dtype=torch.float64, device=pca["mean"].device)
    return ((value - pca["mean"]) @ pca["vectors"]).cpu().numpy()


def reconstruct(pca, x):
    import torch
    value = torch.as_tensor(x, dtype=torch.float64, device=pca["mean"].device)
    return (value @ pca["vectors"].T + pca["mean"]).cpu().numpy()


def fit_dmd(pca, x, y):
    import torch
    a = torch.as_tensor(project(pca, x), dtype=torch.float64, device=pca["mean"].device)
    b = torch.as_tensor(project(pca, y), dtype=torch.float64, device=a.device)
    design = torch.cat([a, torch.ones((len(a), 1), dtype=a.dtype, device=a.device)], dim=1)
    coefficient = torch.linalg.pinv(design) @ b
    return coefficient


def dmd_rollout(pca, coefficient, x, horizons):
    import torch
    state = torch.as_tensor(project(pca, x), dtype=torch.float64, device=coefficient.device)
    predictions = {}
    with torch.inference_mode():
        for k in range(1, max(horizons) + 1):
            state = state @ coefficient[:-1] + coefficient[-1]
            if k in horizons:
                predictions[k] = reconstruct(pca, state.cpu().numpy())
    return predictions


def fit_var(pca, train, val, order, alphas, device):
    a = project(pca, train["lags"][:, -order:])
    b = project(pca, val["lags"][:, -order:])
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    prepared, target = prepare_ridge(a, device), project(pca, train["y1"])
    best, raw_table = None, []
    for alpha in alphas:
        candidate = fit_ridge(prepared, target, alpha)
        prediction = reconstruct(pca, predict_ridge(candidate, b))
        scores = series_metrics(val["y1"], prediction, train["y1"].mean(0), val["ids"])
        score = np.mean([r["r2"] for r in scores])
        raw_table.append({"alpha": alpha, "r2": float(score)})
        if best is None or (score, -alpha) > best[:2]:
            best = score, -alpha, candidate
    return best[2], -best[1], raw_table


def var_rollout(pca, fit, lagged, order, horizons):
    history = project(pca, lagged[:, -order:]).copy()
    predictions = {}
    for k in range(1, max(horizons) + 1):
        state = predict_ridge(fit, history.reshape(len(history), -1))
        history = np.concatenate([history[:, 1:], state[:, None]], axis=1)
        if k in horizons:
            predictions[k] = reconstruct(pca, state)
    return predictions


def series_metrics(target, prediction, reference, ids):
    return [{"series_id": str(series), "n_val": int(np.sum(ids == series)),
             **diagnostic_metrics(target[ids == series], prediction[ids == series], reference)}
            for series in np.unique(ids)]


def coordinate_scores(target, prediction, reference):
    denominator = np.square(target.astype(np.float64) - reference).sum(0)
    valid = denominator > 1e-12
    values = 1 - np.square(target.astype(np.float64) - prediction).sum(0)[valid] / denominator[valid]
    return {"dimensions": int(valid.sum()), "r2_quantiles": dict(zip(
        ["p05", "p25", "median", "p75", "p95"], np.quantile(values, [.05, .25, .5, .75, .95]).tolist())),
        "fraction_dimensions_positive": float(np.mean(values > 0))}


def matched_mlp_config(base, input_width, output_width=768):
    config = dict(base)
    target = base["width"] * (2 * output_width + 1) + output_width
    config["width"] = max(1, round((target - output_width) / (input_width + output_width + 1)))
    return config
