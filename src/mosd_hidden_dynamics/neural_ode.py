import copy
import time

import numpy as np
import torch


def make_field(dimension, width, device):
    model = torch.nn.Sequential(torch.nn.Linear(dimension, width), torch.nn.Tanh(),
                                torch.nn.Linear(width, dimension)).to(device)
    torch.nn.init.normal_(model[-1].weight, std=.001)
    torch.nn.init.zeros_(model[-1].bias)
    return model


def rk4_step(field, state, dt=1.0):
    a = field(state)
    b = field(state + dt * a / 2)
    c = field(state + dt * b / 2)
    d = field(state + dt * c)
    return state + dt * (a + 2 * b + 2 * c + d) / 6


def advance(field, state, duration=1.0, substeps=1, method="rk4"):
    if method not in ("rk4", "euler"):
        raise ValueError("Unknown integration method")
    count = max(1, int(np.ceil(duration * substeps)))
    for _ in range(count):
        state = (rk4_step(field, state, duration / count) if method == "rk4"
                 else state + duration / count * field(state))
    return state


def fit_ode(train, val, cfg, seed, device):
    torch.manual_seed(seed)
    start = time.perf_counter()
    mean, scale = train["x"].mean(0), train["x"].std(0)
    scale = np.where(scale > 1e-8, scale, 1)
    tx = torch.as_tensor((train["trajectory"] - mean) / scale, dtype=torch.float32, device=device)
    vx = torch.as_tensor((val["trajectory"] - mean) / scale, dtype=torch.float32, device=device)
    weight = torch.as_tensor(np.square(scale), dtype=torch.float32, device=device)
    weight = weight / weight.mean()
    model = make_field(train["x"].shape[1], cfg["width"], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    selection_horizon = cfg.get("selection_horizon", cfg["segment_steps"])
    validation_target = torch.as_tensor((val[f"y{selection_horizon}"] - mean) / scale, dtype=torch.float32, device=device)
    reference = torch.as_tensor((train[f"y{selection_horizon}"].mean(0) - mean) / scale, dtype=torch.float32, device=device)
    masks = [torch.as_tensor(np.flatnonzero(val["ids"] == s), device=device) for s in np.unique(val["ids"])]
    denominator = [((validation_target[mask] - reference).square() * weight).sum() for mask in masks]
    best, best_epoch, saved, history = float("inf"), 0, None, []
    for epoch in range(cfg["epochs"]):
        model.train()
        for indices in torch.randperm(len(tx), device=device).split(cfg["batch_size"]):
            state = tx[indices, 0]
            loss = 0
            for step in range(1, cfg["segment_steps"] + 1):
                state = advance(model, state, substeps=cfg["rk4_substeps"], method=cfg.get("integration_method", "rk4"))
                loss = loss + ((state - tx[indices, step]).square() * weight).mean() / cfg["segment_steps"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite Neural ODE training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            state = vx[:, 0]
            for _ in range(selection_horizon):
                state = advance(model, state, substeps=cfg["rk4_substeps"], method=cfg.get("integration_method", "rk4"))
            error = (state - validation_target).square() * weight
            score = float(torch.stack([error[mask].sum() / den for mask, den in zip(masks, denominator)]).mean())
        history.append({"epoch": epoch + 1, "validation_one_minus_macro_r2_k4": score})
        if np.isfinite(score) and score < best:
            best, best_epoch, saved = score, epoch + 1, copy.deepcopy(model.state_dict())
        if epoch + 1 - best_epoch >= cfg["patience"]:
            break
    if saved is None:
        raise FloatingPointError("No finite Neural ODE validation checkpoint")
    model.load_state_dict(saved)
    return {"model": model.eval(), "mean": mean, "scale": scale, "config": cfg, "seed": seed,
            "best_epoch": best_epoch, "history": history, "seconds": time.perf_counter() - start,
            "parameters": sum(p.numel() for p in model.parameters())}


def save_ode(path, fit):
    torch.save({key: value for key, value in fit.items() if key != "model"} |
               {"state_dict": fit["model"].state_dict()}, path)


def load_ode(path, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    model = make_field(len(saved["mean"]), saved["config"]["width"], device)
    model.load_state_dict(saved.pop("state_dict"))
    saved["model"] = model.eval()
    return saved


def rollout_ode(fit, x, horizons, substeps=None):
    model = fit["model"]
    device = next(model.parameters()).device
    substeps = fit["config"]["rk4_substeps"] if substeps is None else substeps
    result = {}
    with torch.inference_mode():
        state = torch.as_tensor((x - fit["mean"]) / fit["scale"], dtype=torch.float32, device=device)
        for step in range(1, max(horizons) + 1):
            state = advance(model, state, substeps=substeps, method=fit["config"].get("integration_method", "rk4"))
            if step in horizons:
                value = state.cpu().numpy().astype(np.float64) * fit["scale"] + fit["mean"]
                if not np.isfinite(value).all():
                    raise FloatingPointError(f"Nonfinite Neural ODE rollout at step {step}")
                result[step] = value
    return result


def field_diagnostics(fit, train, val, cfg):
    model = fit["model"]
    device = next(model.parameters()).device
    mean = torch.as_tensor(fit["mean"], device=device, dtype=torch.float32)
    scale = torch.as_tensor(fit["scale"], device=device, dtype=torch.float32)
    training = torch.as_tensor((train - fit["mean"]) / fit["scale"], device=device, dtype=torch.float32)
    rng = np.random.default_rng(fit["seed"])
    chosen = rng.choice(len(training), size=min(cfg["root_starts"], len(training)), replace=False)
    starts = training[chosen].detach().clone()
    with torch.no_grad():
        magnitude = (model(training) * scale).abs().amax(dim=1).median()
    threshold = float(magnitude) * cfg["root_tolerance_fraction"]
    starts[0] = training.mean(0)
    point = torch.nn.Parameter(starts)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.LBFGS([point], max_iter=cfg["root_iterations"], history_size=30,
                                 tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = ((model(point) * scale) / magnitude.clamp_min(1e-8)).square().mean()
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        residual = (model(point) * scale).abs().amax(dim=1).cpu().numpy()
        candidates = (point * scale + mean).detach().cpu().numpy()
    unique, spectra, candidate_rows = [], [], []
    cloud = np.concatenate([train, val]).astype(np.float64)
    cloud_norm = np.linalg.norm(cloud, axis=1).clip(1e-12)
    for index, (raw, error) in enumerate(zip(candidates, residual)):
        row = {"start": index, "field_infinity_norm": float(error), "threshold": threshold,
               "accepted": bool(error < threshold)}
        if row["accepted"]:
            distances = np.linalg.norm(cloud - raw, axis=1) / cloud_norm
            row["relative_distance_to_nearest_observed_state"] = float(distances.min())
            duplicate = next((j for j, old in enumerate(unique)
                              if np.linalg.norm(raw - old) / max(np.linalg.norm(old), 1e-12) < .01), None)
            if duplicate is None:
                jacobian = torch.func.jacrev(model)(point[index].detach())
                eigenvalues = torch.linalg.eigvals(jacobian).detach().cpu().numpy()
                maximum = float(eigenvalues.real.max())
                row.update(root=len(unique), largest_real_eigenvalue=maximum,
                           classification="unstable" if maximum > 1e-6 else "stable" if maximum < -1e-6 else "nonhyperbolic")
                unique.append(raw)
                spectra.append(eigenvalues)
            else:
                row["duplicate_of_root"] = duplicate
        candidate_rows.append(row)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return {"median_on_data_field_infinity_norm": float(magnitude), "threshold": threshold,
            "distinct_approximate_zeros": len(unique), "root_cluster_relative_distance": .01,
            "candidates": candidate_rows}, np.asarray(unique), np.asarray(spectra)
