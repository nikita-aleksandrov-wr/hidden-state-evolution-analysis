import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Frozen-field solver refinement at short, fractional and long horizons")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "4"
    import numpy as np
    import torch
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mosd_hidden_dynamics.extended import load_extended_group, series_metrics
    from mosd_hidden_dynamics.neural_ode import advance, load_ode, rollout_ode
    from mosd_hidden_dynamics.study import write_rows
    from mosd_hidden_dynamics.utils import save_json, source_record
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(args.run_dir).resolve()
    cfg = json.loads((root / "config_requested.json").read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    output = root / "ode_resolution"
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
            group, anchors, digest = load_extended_group(cfg, registry, root, name, protocol, order=8, segment=4)
            train, val = group["train"], group["val"]
            dense_dir = root / "ode_interpolation" / f"{name}_{protocol}"
            dense_truth = []
            for record in registry:
                eligible = [a for a in anchors if a["split"] == "val" and a["series_id"] == record["series_id"]]
                if not eligible:
                    continue
                with np.load(dense_dir / (record["series_id"] + "_dense.npz")) as data:
                    ends = np.asarray([a["source_end"] for a in eligible])[:, None] + np.arange(17)[None] * 4
                    indices = np.searchsorted(data["raw_endpoints"], ends)
                    np.testing.assert_array_equal(data["raw_endpoints"][indices], ends)
                    dense_truth.append(data["states"][indices])
            dense_truth = np.concatenate(dense_truth)
            long_group, long_digest = None, None
            if protocol == "out_of_series":
                long_cfg = copy.deepcopy(cfg)
                long_cfg["horizons"] = [1, 4, 8, 16, 32, 64, 128, 256]
                long_group, _, long_digest = load_extended_group(long_cfg, registry, root, name, protocol, order=8, segment=8)
                expected = json.loads((root / "ode_full_rollouts" / name / "run_metadata.json").read_text())
                assert expected["anchor_sha256"] == long_digest
            rows, changes = [], []
            common = {"dataset": name, "family": dataset["family"], "protocol": protocol, "model": "neural_ode"}
            for seed in (2026, 2027, 2028):
                fit = load_ode(root / "neural_ode" / f"{name}_{protocol}/ode_seed{seed}.pt", args.device)
                short = {}
                for substeps in (1, 2, 4, 8, 16):
                    short[substeps] = rollout_ode(fit, val["x"], [1, 4], substeps=substeps)
                    for k, prediction in short[substeps].items():
                        rows.extend({**common, "section": "short", "substeps": substeps, "horizon": k, "seed": seed, **r}
                                    for r in series_metrics(val[f"y{k}"], prediction, train[f"y{k}"].mean(0), val["ids"]))
                for substeps in (1, 2, 4, 8):
                    for k in (1, 4):
                        changes.append({"seed": seed, "substeps": substeps, "horizon": k,
                            "relative_l2_difference_vs_16": float(np.linalg.norm(short[substeps][k] - short[16][k])
                                                                 / np.linalg.norm(short[16][k]))})
                for substeps in (4, 16):
                    with torch.inference_mode():
                        state = torch.as_tensor((val["x"] - fit["mean"]) / fit["scale"], device=args.device, dtype=torch.float32)
                        predictions = [val["x"]]
                        for _ in range(16):
                            state = advance(fit["model"], state, duration=.25, substeps=substeps)
                            predictions.append(state.cpu().numpy().astype(np.float64) * fit["scale"] + fit["mean"])
                    prediction = np.stack(predictions, axis=1)
                    for subset, indices in (("all_future", np.arange(1, 17)),
                                            ("interior", np.flatnonzero(np.arange(17) % 4))):
                        truth = dense_truth[:, indices].reshape(-1, 768)
                        predicted = prediction[:, indices].reshape(-1, 768)
                        rows.extend({**common, "section": "dense", "subset": subset, "substeps": substeps,
                                     "horizon": 4, "seed": seed, **r}
                                    for r in series_metrics(truth, predicted, train["x"].mean(0),
                                                            np.repeat(val["ids"], len(indices))))
                    if long_group is not None:
                        for k, predicted in rollout_ode(fit, long_group["val"]["x"], long_cfg["horizons"], substeps=substeps).items():
                            rows.extend({**common, "section": "long", "substeps": substeps, "horizon": k,
                                         "seed": seed, **r} for r in series_metrics(long_group["val"][f"y{k}"],
                                predicted, long_group["train"][f"y{k}"].mean(0), long_group["val"]["ids"]))
                print("REFINED", name, protocol, seed, flush=True)
                save_json(directory / "metrics.json", rows)
            save_json(directory / "changes.json", changes)
            write_rows(directory / "metrics.csv", rows)
            save_json(directory / "run_metadata.json", {"source": source, "anchor_sha256": digest,
                "long_anchor_sha256": long_digest, "refitted": False, "short_substeps": [1, 2, 4, 8, 16],
                "dense_long_substeps": [4, 16], "test_evaluated": False})
            save_json(directory / "complete.json", {"seconds": time.perf_counter() - started, "test_evaluated": False})
            del group, train, val, long_group
            torch.cuda.empty_cache()
    rows = [r for p in output.glob("*/metrics.json") for r in json.loads(p.read_text())]
    save_json(output / "metrics.json", rows)
    save_json(output / "complete.json", {"groups": 6, "refitted": False, "test_evaluated": False})


if __name__ == "__main__":
    main()
