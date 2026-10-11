import time

import numpy as np

from .utils import save_json, sha256_file


def generate_lorenz(cfg, seed):
    from dysts.flows import Lorenz

    if cfg["system"] != "Lorenz" or cfg["noise_std"] != 0:
        raise ValueError("This checkpoint supports deterministic, noise-free Lorenz only")
    model = Lorenz()
    model.ic = np.asarray(cfg["initial_condition"], dtype=np.float64)
    initial_condition = model.ic.copy()
    n = cfg["trajectory_length"] + cfg["burn_in"]
    times, trajectory = model.make_trajectory(
        n,
        resample=True,
        pts_per_period=cfg["points_per_period"],
        return_times=True,
        standardize=False,
        postprocess=False,
        noise=0.0,
        method="DOP853",
        random_seed=seed,
        rtol=cfg["rtol"],
        atol=cfg["atol"],
    )
    if trajectory is None or trajectory.shape != (n, 3):
        raise RuntimeError("Lorenz integration did not return a complete three-dimensional trajectory")
    trajectory = np.asarray(trajectory[cfg["burn_in"]:], dtype=np.float64)
    times = np.asarray(times[cfg["burn_in"]:], dtype=np.float64)
    if not np.isfinite(trajectory).all():
        raise ValueError("Non-finite generated data")
    observed = trajectory[:, cfg["coordinate"]].astype(np.float32)
    metadata = {
        "family": "chaotic", "system": "Lorenz", "series_id": "lorenz_0",
        "parameters": {k: float(v) for k, v in model.params.items()},
        "initial_condition": initial_condition.tolist(),
        "internal_dt": float(model.dt), "dominant_period": float(model.period),
        "sampling_dt": float(times[1] - times[0]),
        "generation": cfg, "seed": seed, "method": "DOP853",
        "standardize": False, "postprocess": False, "noise": 0.0,
        "input": "univariate coordinate", "coordinate": cfg["coordinate"],
        "shape": list(trajectory.shape),
    }
    return observed, trajectory, times, metadata


def prepare_data(cfg, run_dir):
    start = time.perf_counter()
    observed, trajectory, times, metadata = generate_lorenz(cfg["data"], cfg["seed"])
    path = run_dir / "data.npz"
    np.savez_compressed(path, observed=observed, trajectory=trajectory, times=times)
    metadata["file_sha256"] = sha256_file(path)
    save_json(run_dir / "data_metadata.json", metadata)
    return observed, metadata, time.perf_counter() - start
