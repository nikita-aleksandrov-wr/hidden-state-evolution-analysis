import time

import numpy as np

from .baselines import prediction_metrics


def prepare_ridge(x, device):
    import torch

    x = torch.as_tensor(x, dtype=torch.float64, device=device)
    mean = x.mean(dim=0)
    scale = x.std(dim=0, correction=0)
    scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
    z = (x - mean) / scale
    z_mean = z.mean(dim=0)
    z = z - z_mean
    eigenvalues, eigenvectors = torch.linalg.eigh(z.T @ z)
    return {"x": z, "mean": mean, "scale": scale, "z_mean": z_mean,
            "values": eigenvalues.clamp_min(0), "vectors": eigenvectors}


def ridge_statistics(prepared, y):
    import torch

    y = torch.as_tensor(y, dtype=torch.float64, device=prepared["x"].device)
    mean = y.mean(dim=0)
    q = prepared["vectors"]
    projected = q.T @ (prepared["x"].T @ (y - mean))
    return mean, projected


def fit_ridge(prepared, y, alpha, statistics=None):
    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive")
    mean, projected = ridge_statistics(prepared, y) if statistics is None else statistics
    q = prepared["vectors"]
    coefficient = (q @ (projected / (prepared["values"][:, None] + alpha))).T
    return {"coef": coefficient, "intercept": mean - coefficient @ prepared["z_mean"],
            "mean": prepared["mean"], "scale": prepared["scale"]}


def predict_ridge(model, x):
    import torch

    with torch.inference_mode():
        x = torch.as_tensor(x, dtype=torch.float64, device=model["coef"].device)
        return (((x - model["mean"]) / model["scale"]) @ model["coef"].T
                + model["intercept"]).cpu().numpy()


def select_ridge(prepared, y, x_val, y_val, alphas, series_ids=None):
    reference = np.asarray(y, dtype=np.float64).mean(axis=0)
    best, table = None, []
    statistics = ridge_statistics(prepared, y)
    for alpha in alphas:
        model = fit_ridge(prepared, y, alpha, statistics)
        prediction = predict_ridge(model, x_val)
        scores = prediction_metrics(y_val, prediction, reference)
        if series_ids is not None:
            individual = [prediction_metrics(y_val[series_ids == series], prediction[series_ids == series], reference)
                          for series in np.unique(series_ids)]
            scores = {metric: float(np.mean([row[metric] for row in individual])) for metric in ("r2", "cosine")}
        table.append({"alpha": alpha, **scores})
        if best is None or (scores["r2"], -alpha) > (best[0], -best[1]):
            best = (scores["r2"], alpha, model)
    return best[2], best[1], table


def rollout_ridge(model, x, horizons):
    import torch

    predictions = {}
    with torch.inference_mode():
        state = torch.as_tensor(x, dtype=torch.float64, device=model["coef"].device)
        for step in range(1, max(horizons) + 1):
            state = ((state - model["mean"]) / model["scale"]) @ model["coef"].T + model["intercept"]
            if step in horizons:
                predictions[step] = state.cpu().numpy().copy()
    return predictions


def diagnostic_metrics(target, prediction, train_mean):
    scores = prediction_metrics(target, prediction, train_mean)
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    a, b = target - train_mean, prediction - train_mean
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    scores["centered_cosine"] = (float(np.mean(np.clip(np.sum(a * b, axis=1) / norms, -1, 1)))
                                 if np.all(norms > 1e-12) else None)
    denominator = np.square(target - target.mean(axis=0)).sum()
    scores["r2_validation_mean"] = float(1 - np.square(target - prediction).sum() / denominator) if denominator > 0 else None
    scores["relative_error"] = float(np.linalg.norm(target - prediction) / np.linalg.norm(target))
    scores["prediction_norm_ratio"] = float(np.linalg.norm(prediction) / np.linalg.norm(target))
    return scores


def save_ridge(path, model):
    np.savez_compressed(path, **{name: value.detach().cpu().numpy() for name, value in model.items()})


def fit_mlp(x, y, x_val, y_val, cfg, seed, device, series_ids=None):
    import copy
    import torch

    torch.manual_seed(seed)
    start = time.perf_counter()
    x_mean, x_scale = x.mean(axis=0), x.std(axis=0)
    x_scale = np.where(x_scale > 1e-8, x_scale, 1)
    y_mean, y_scale = y.mean(axis=0), y.std(axis=0)
    y_scale = np.where(y_scale > 1e-8, y_scale, 1)
    tx = torch.as_tensor((x - x_mean) / x_scale, dtype=torch.float32, device=device)
    ty = torch.as_tensor((y - y_mean) / y_scale, dtype=torch.float32, device=device)
    vx = torch.as_tensor((x_val - x_mean) / x_scale, dtype=torch.float32, device=device)
    vy = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    ym = torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    ys = torch.as_tensor(y_scale, dtype=torch.float32, device=device)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], cfg["width"]), torch.nn.GELU(),
                                torch.nn.Linear(cfg["width"], y.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best_loss, best_epoch, best_state, history = float("inf"), 0, None, []
    weights = ys.square()
    weights = weights / weights.mean()
    selections = ([torch.as_tensor(np.flatnonzero(series_ids == series), device=device)
                   for series in np.unique(series_ids)] if series_ids is not None else None)
    denominators = [(vy[index] - ym).square().sum() for index in selections] if selections is not None else None
    for epoch in range(cfg["epochs"]):
        model.train()
        order = torch.randperm(len(tx), device=device)
        for batch in order.split(cfg["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(tx[batch]) - ty[batch]).square() * weights).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            error = (model(vx) * ys + ym - vy).square()
            validation_loss = (float(torch.stack([error[index].sum() / denominator
                                                 for index, denominator in zip(selections, denominators)]).mean())
                               if selections is not None else float(error.mean()))
        history.append({"epoch": epoch + 1, "validation_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch, best_state = validation_loss, epoch + 1, copy.deepcopy(model.state_dict())
        if epoch + 1 - best_epoch >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "x_mean": x_mean, "x_scale": x_scale,
            "y_mean": y_mean, "y_scale": y_scale, "seed": seed,
            "best_epoch": best_epoch, "history": history,
            "seconds": time.perf_counter() - start,
            "parameters": sum(p.numel() for p in model.parameters())}


def predict_mlp(fit, x):
    import torch

    device = next(fit["model"].parameters()).device
    with torch.inference_mode():
        x = torch.as_tensor((x - fit["x_mean"]) / fit["x_scale"], dtype=torch.float32, device=device)
        prediction = fit["model"](x).cpu().numpy().astype(np.float64)
    return prediction * fit["y_scale"] + fit["y_mean"]


def rollout_mlp(fit, x, horizons):
    predictions = {}
    state = x.copy()
    for step in range(1, max(horizons) + 1):
        state = predict_mlp(fit, state)
        if step in horizons:
            predictions[step] = state.copy()
    return predictions


def load_ridge(path, device):
    import torch
    with np.load(path) as data:
        return {name: torch.as_tensor(data[name], dtype=torch.float64, device=device) for name in data.files}


def load_mlp(path, device):
    import torch
    # These are this study's own checkpoints, including their fitted NumPy scalers.
    saved = torch.load(path, map_location=device, weights_only=False)
    model = torch.nn.Sequential(torch.nn.Linear(len(saved["x_mean"]), saved["config"]["width"]), torch.nn.GELU(),
                                torch.nn.Linear(saved["config"]["width"], len(saved["y_mean"]))).to(device)
    model.load_state_dict(saved["state_dict"])
    result = {"model": model.eval()}
    for name in ("x_mean", "x_scale", "y_mean", "y_scale"):
        value = saved[name]
        result[name] = value.cpu().numpy() if isinstance(value, torch.Tensor) else value
    return result


def nearest_indices(train, val, k, device):
    import torch
    def normalize(x):
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        x = x - x.mean(dim=1, keepdim=True)
        return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    train = normalize(train)
    all_indices, all_cosines = [], []
    for start in range(0, len(val), 256):
        similarity = normalize(val[start:start + 256]) @ train.T
        values, indices = torch.topk(similarity, k, dim=1)
        all_indices.append(indices.cpu().numpy())
        all_cosines.append(values.clamp(-1, 1).cpu().numpy())
    return np.concatenate(all_indices), np.concatenate(all_cosines)
