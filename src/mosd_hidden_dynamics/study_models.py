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


def fit_ridge(prepared, y, alpha):
    import torch

    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive")
    y = torch.as_tensor(y, dtype=torch.float64, device=prepared["x"].device)
    mean = y.mean(dim=0)
    q = prepared["vectors"]
    projected = q.T @ (prepared["x"].T @ (y - mean))
    coefficient = (q @ (projected / (prepared["values"][:, None] + alpha))).T
    return {"coef": coefficient, "intercept": mean - coefficient @ prepared["z_mean"],
            "mean": prepared["mean"], "scale": prepared["scale"]}


def predict_ridge(model, x):
    import torch

    with torch.inference_mode():
        x = torch.as_tensor(x, dtype=torch.float64, device=model["coef"].device)
        return (((x - model["mean"]) / model["scale"]) @ model["coef"].T
                + model["intercept"]).cpu().numpy()


def select_ridge(prepared, y, x_val, y_val, alphas):
    reference = np.asarray(y, dtype=np.float64).mean(axis=0)
    best, table = None, []
    for alpha in alphas:
        model = fit_ridge(prepared, y, alpha)
        scores = prediction_metrics(y_val, predict_ridge(model, x_val), reference)
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


def fit_mlp(x, y, x_val, y_val, cfg, seed, device):
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
            validation_loss = float((model(vx) * ys + ym - vy).square().mean())
        history.append({"epoch": epoch + 1, "validation_mse": validation_loss})
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
