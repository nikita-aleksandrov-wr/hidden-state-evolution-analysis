import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def prediction_metrics(target, prediction, train_target_mean):
    target, prediction = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("Targets and predictions must be matching two-dimensional arrays")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Non-finite targets or predictions")
    denominator = np.square(target - train_target_mean).sum()
    norms = np.linalg.norm(target, axis=1) * np.linalg.norm(prediction, axis=1)
    if denominator <= 0 or np.any(norms == 0):
        raise ValueError("R2 or cosine is undefined for these targets/predictions")
    cosine = np.sum(target * prediction, axis=1) / norms
    return {"r2": float(1 - np.square(target - prediction).sum() / denominator),
            "cosine": float(np.clip(cosine, -1, 1).mean())}


def get_pairs(rows, states, endpoints, split):
    if split not in ("train", "val"):
        raise ValueError("The checkpoint pipeline does not evaluate test states")
    lookup = {int(e): i for i, e in enumerate(endpoints)}
    selected = [r for r in rows if r["eligible"] and r["split"] == split]
    x = states[[lookup[r["raw_context_end"]] for r in selected]].astype(np.float64)
    y = states[[lookup[r["target_context_end"]] for r in selected]].astype(np.float64)
    return x, y


def fit_baselines(cfg, rows, run_dir):
    with np.load(run_dir / "states.npz") as data:
        states, endpoints = data["states"][:, 0], data["raw_endpoints"]
    x_train, y_train = get_pairs(rows, states, endpoints, "train")
    x_val, y_val = get_pairs(rows, states, endpoints, "val")
    mean = y_train.mean(axis=0)
    metrics = {
        "train_mean": prediction_metrics(y_val, np.broadcast_to(mean, y_val.shape), mean),
        "persistence": prediction_metrics(y_val, x_val, mean),
    }
    scaler = None
    if cfg["ridge"]["scale_inputs"]:
        scaler = StandardScaler().fit(x_train)
        x_train, x_val = scaler.transform(x_train), scaler.transform(x_val)
    if cfg["ridge"]["selection_metric"] != "validation_r2":
        raise ValueError("Only the predeclared validation_r2 selection rule is supported")
    alpha_table, best, best_model = [], None, None
    fit_seconds, prediction_seconds = 0.0, 0.0
    for alpha in cfg["ridge"]["alpha_grid"]:
        if alpha <= 0:
            raise ValueError("Ridge alpha must be positive")
        start = time.perf_counter()
        model = Ridge(alpha=alpha, solver="svd").fit(x_train, y_train)
        fit_seconds += time.perf_counter() - start
        start = time.perf_counter()
        prediction = model.predict(x_val)
        prediction_seconds += time.perf_counter() - start
        result = {"alpha": alpha, **prediction_metrics(y_val, prediction, mean)}
        alpha_table.append(result)
        if best is None or (result["r2"], -alpha) > (best["r2"], -best["alpha"]):
            best, best_model = result, model
    metrics["ridge"] = {k: best[k] for k in ("r2", "cosine")}
    if cfg["ridge"]["shuffled_target_control"]:
        permutation = np.random.default_rng(cfg["seed"]).permutation(len(y_train))
        start = time.perf_counter()
        control = Ridge(alpha=best["alpha"], solver="svd").fit(x_train, y_train[permutation])
        fit_seconds += time.perf_counter() - start
        start = time.perf_counter()
        prediction = control.predict(x_val)
        prediction_seconds += time.perf_counter() - start
        metrics["shuffled_target_ridge"] = prediction_metrics(y_val, prediction, mean)
    np.savez_compressed(
        run_dir / "ridge_fit.npz", coef=best_model.coef_, intercept=best_model.intercept_,
        input_mean=scaler.mean_ if scaler is not None else np.zeros(x_train.shape[1]),
        input_scale=scaler.scale_ if scaler is not None else np.ones(x_train.shape[1]),
        train_target_mean=mean,
    )
    return {
        "split": "validation", "preliminary": True, "test_evaluated": False,
        "n_train_pairs": len(y_train), "n_val_pairs": len(y_val),
        "baselines": metrics, "ridge_alpha_grid": alpha_table, "selected_alpha": best["alpha"],
        "selection_rule": "maximum validation R2; ties use the smaller alpha",
        "r2_reference": "training-target mean; vector aggregate",
        "cosine_aggregation": "arithmetic mean over validation pairs",
        "scaler_fitted_split": "train", "targets_scaled": False,
        "shuffled_control_alpha": best["alpha"] if cfg["ridge"]["shuffled_target_control"] else None,
    }, {"fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds}
