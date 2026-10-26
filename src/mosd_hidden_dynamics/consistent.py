import copy
import time

import numpy as np
import torch

from .neural_ode import make_field, rk4_step
from .study_models import diagnostic_metrics


def transition(field, state, method, substeps=4):
    if method == "mlp":
        return field(state)
    if method == "residual":
        return state + field(state)
    if method != "ode" or substeps < 1:
        raise ValueError("Unknown transition or invalid integration resolution")
    for _ in range(substeps):
        state = rk4_step(field, state, 1 / substeps)
    return state


def fit_transition(train, val, cfg, method, seed, device, stop_path=None):
    torch.manual_seed(seed)
    started = time.perf_counter()
    mean, scale = train["x"].mean(0), train["x"].std(0)
    scale = np.where(scale > 1e-8, scale, 1)
    trajectories = torch.as_tensor((train["trajectory"] - mean) / scale, device=device, dtype=torch.float32)
    initial = torch.as_tensor((val["x"] - mean) / scale, device=device, dtype=torch.float32)
    target = torch.as_tensor((val["y4"] - mean) / scale, device=device, dtype=torch.float32)
    reference = torch.as_tensor((train["y4"].mean(0) - mean) / scale, device=device)
    weights = torch.as_tensor(scale ** 2, device=device)
    weights = weights / weights.mean()
    masks = [torch.as_tensor(np.flatnonzero(val["ids"] == sid), device=device) for sid in np.unique(val["ids"])]
    denominators = [((target[m] - reference).square() * weights).sum() for m in masks]
    field = make_field(len(mean), cfg["width"], device)
    optimizer = torch.optim.AdamW(field.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    best, best_epoch, saved, history = float("inf"), 0, None, []
    for epoch in range(cfg["epochs"]):
        field.train()
        for indices in torch.randperm(len(trajectories), device=device).split(cfg["batch_size"]):
            if stop_path is not None and stop_path.exists():
                raise InterruptedError("Stop requested; completed fits remain saved; unfinished fit restarts from its seed")
            state = trajectories[indices, 0]
            loss = 0
            for step in range(1, cfg["segment_steps"] + 1):
                state = transition(field, state, method, cfg["rk4_substeps"])
                loss = loss + ((state - trajectories[indices, step]).square() * weights).mean() / cfg["segment_steps"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite transition training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(field.parameters(), 1)
            optimizer.step()
        field.eval()
        with torch.inference_mode():
            state = initial
            for _ in range(cfg["selection_horizon"]):
                state = transition(field, state, method, cfg["rk4_substeps"])
            errors = (state - target).square() * weights
            value = float(torch.stack([errors[m].sum() / den for m, den in zip(masks, denominators)]).mean())
        history.append({"epoch": epoch + 1, "validation_one_minus_macro_r2_k4": value})
        if np.isfinite(value) and value < best:
            best, best_epoch, saved = value, epoch + 1, copy.deepcopy(field.state_dict())
        if epoch + 1 - best_epoch >= cfg["patience"]:
            break
    if saved is None:
        raise FloatingPointError("No finite validation checkpoint")
    field.load_state_dict(saved)
    return {"model": field.eval(), "mean": mean, "scale": scale, "config": cfg, "method": method,
            "seed": seed, "best_epoch": best_epoch, "history": history,
            "seconds": time.perf_counter() - started, "parameters": sum(p.numel() for p in field.parameters())}


def save_transition(path, fit):
    torch.save({k: v for k, v in fit.items() if k != "model"} | {"state_dict": fit["model"].state_dict()}, path)


def load_transition(path, device):
    fit = torch.load(path, map_location=device, weights_only=False)
    field = make_field(len(fit["mean"]), fit["config"]["width"], device)
    field.load_state_dict(fit.pop("state_dict"))
    fit["model"] = field.eval()
    return fit


def scoring_space(x, device):
    values = torch.as_tensor(x, device=device, dtype=torch.float64)
    mean = values.mean(0)
    centered = values - mean
    covariance = centered.T @ centered / len(values)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues, eigenvectors = eigenvalues.flip(0).clamp_min(0), eigenvectors.flip(1)
    variance = covariance.diag()
    valid = variance > max(float(variance.max()) * 1e-8, 1e-12)
    rank = int(torch.searchsorted(eigenvalues.cumsum(0) / eigenvalues.sum(), .95)) + 1
    scale = variance.sqrt().cpu().numpy()
    return {"mean": mean.cpu().numpy(), "scale": np.where(scale > 1e-8, scale, 1),
            "valid": valid.cpu().numpy(), "vectors": eigenvectors[:, :rank].cpu().numpy(),
            "pc_scale": eigenvalues[:rank].sqrt().clamp_min(1e-8).cpu().numpy(),
            "coordinate_variance": variance.cpu().numpy(), "pc_variance": eigenvalues.cpu().numpy(),
            "rank": rank}


def transformed_r2(target, prediction, reference):
    denominator = np.square(target - reference).sum()
    return float(1 - np.square(target - prediction).sum() / denominator) if denominator > 1e-12 else None


def score_series(target, prediction, reference, ids, space):
    rows = []
    valid, scale = space["valid"], space["scale"]
    vectors, pc_scale = space["vectors"], space["pc_scale"]
    for sid in np.unique(ids):
        mask = ids == sid
        a, b = target[mask].astype(np.float64), prediction[mask].astype(np.float64)
        scores = diagnostic_metrics(a, b, reference)
        scores["r2_standardized"] = transformed_r2(a[:, valid] / scale[valid], b[:, valid] / scale[valid], reference[valid] / scale[valid])
        scores["r2_whitened_pca95"] = transformed_r2(a @ vectors / pc_scale, b @ vectors / pc_scale, reference @ vectors / pc_scale)
        rows.append({"series_id": str(sid), "n_val": int(mask.sum()), **scores})
    return rows


def fixed_starts(ids, count, seed=2026):
    rng = np.random.default_rng(seed)
    return np.concatenate([np.sort(rng.choice(np.flatnonzero(ids == sid), min(count, np.sum(ids == sid)), replace=False))
                           for sid in np.unique(ids)])


def projected_paths(root, anchors, indices, vectors, mean, horizon):
    paths = []
    for index in indices:
        record = anchors[index]
        with np.load(root / "features/pretrained" / (record["series_id"] + ".npz")) as data:
            start = int(np.searchsorted(data["raw_endpoints"], record["source_end"]))
            assert data["raw_endpoints"][start] == record["source_end"]
            values = data["states"][start:start + horizon + 1, 0].astype(np.float64)
        assert len(values) == horizon + 1
        paths.append((values - mean) @ vectors)
    return np.asarray(paths)


def path_statistics(predicted, observed, train_pc_scale):
    # These are finite-rollout statistics, not a recovered-attractor claim.
    rng = np.random.default_rng(2026)
    directions = rng.normal(size=(predicted.shape[-1], 32))
    directions /= np.linalg.norm(directions, axis=0)
    a = (predicted[:, -128:] / train_pc_scale).reshape(-1, predicted.shape[-1])
    b = (observed[:, -128:] / train_pc_scale).reshape(-1, observed.shape[-1])
    sliced = float(np.sqrt(np.mean((np.sort(a @ directions, axis=0) - np.sort(b @ directions, axis=0)) ** 2)))
    def temporal(values):
        values = values[:, 1:, :2]
        values = values - values.mean(axis=1, keepdims=True)
        denominator = np.square(values).sum(axis=1)
        acf = np.stack([np.divide((values[:, :-lag] * values[:, lag:]).sum(axis=1), denominator,
                                 out=np.zeros_like(denominator), where=denominator > 1e-12)
                        for lag in (1, 4, 16, 64)], axis=-1)
        power = np.square(np.abs(np.fft.rfft(values, axis=1))).mean(axis=0)
        power = np.divide(power, power.sum(axis=0, keepdims=True), out=np.zeros_like(power), where=power.sum(axis=0, keepdims=True) > 1e-12)
        return acf.mean(axis=0), power, float(np.mean(denominator <= 1e-12))
    acf_a, psd_a, constant = temporal(predicted)
    acf_b, psd_b, _ = temporal(observed)
    return {"sliced_w2_pc8_tail128": sliced, "acf_mae_pc2": float(np.abs(acf_a - acf_b).mean()),
            "psd_relative_l2_pc2": float(np.linalg.norm(psd_a - psd_b) / max(np.linalg.norm(psd_b), 1e-12)),
            "constant_projected_trajectory_fraction": constant}
