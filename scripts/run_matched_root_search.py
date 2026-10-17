import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Search frozen fields from 50 fixed observed starts")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "4"
    import numpy as np
    import torch
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mosd_hidden_dynamics.extended import load_extended_group
    from mosd_hidden_dynamics.neural_ode import field_diagnostics, load_ode
    from mosd_hidden_dynamics.utils import save_json, source_record
    torch.set_num_threads(4)
    root = Path(args.run_dir).resolve()
    cfg = json.loads((root / "config_requested.json").read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    settings = json.loads((root / "neural_ode/config_requested.json").read_text())["ode"]
    settings = dict(settings, root_starts=50)
    output = root / "matched_roots"
    output.mkdir(exist_ok=True)
    source = source_record(repo)
    for dataset in cfg["datasets"]:
        if dataset["family"] == "control":
            continue
        name = dataset["name"]
        for protocol in ("in_series", "out_of_series"):
            directory = output / f"{name}_{protocol}"
            if (directory / "complete.json").exists():
                continue
            directory.mkdir(exist_ok=True)
            started = time.perf_counter()
            group, _, digest = load_extended_group(cfg, registry, root, name, protocol, order=8, segment=4)
            if protocol == "out_of_series":
                long_cfg = copy.deepcopy(cfg)
                long_cfg["horizons"] = [1, 4, 8, 16, 32, 64, 128, 256]
                long, _, _ = load_extended_group(long_cfg, registry, root, name, protocol, order=8, segment=8)
                with np.load(root / "ode_full_rollouts" / name / "trajectories.npz") as data:
                    indices = data["initial_indices"]
                starts = long["val"]["x"][indices].copy()
                del long
            else:
                rng = np.random.default_rng(cfg["seed"])
                ids = group["val"]["ids"]
                selected = []
                for series in np.unique(ids):
                    choices = np.flatnonzero(ids == series)
                    selected.extend(rng.choice(choices, min(len(choices), int(np.ceil(50 / len(np.unique(ids))))), replace=False))
                starts = group["val"]["x"][np.asarray(selected[:50])].copy()
            np.savez_compressed(directory / "initial_states.npz", states=starts)
            for seed in (2026, 2027, 2028):
                fit = load_ode(root / "neural_ode" / f"{name}_{protocol}/ode_seed{seed}.pt", args.device)
                diagnostics, roots, spectra = field_diagnostics(fit, group["train"]["x"], group["val"]["x"],
                                                               settings, initial_states=starts)
                save_json(directory / f"roots_seed{seed}.json", diagnostics)
                np.savez_compressed(directory / f"roots_seed{seed}.npz", roots=roots, eigenvalues=spectra)
                print("MATCHED ROOTS", name, protocol, seed, diagnostics["distinct_approximate_zeros"], flush=True)
            save_json(directory / "run_metadata.json", {"source": source, "anchor_sha256": digest, "starts": 50,
                "same_starts_as_long_rollouts": protocol == "out_of_series", "settings": settings,
                "refitted": False, "test_evaluated": False})
            save_json(directory / "complete.json", {"seconds": time.perf_counter() - started, "test_evaluated": False})
    save_json(output / "complete.json", {"groups": 6, "test_evaluated": False})


if __name__ == "__main__":
    main()
