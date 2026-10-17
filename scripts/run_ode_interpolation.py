import argparse
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Held-out quarter-step ODE interpolation against discrete ridge bridges")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mosd_hidden_dynamics.extraction import extract_vectors, load_backbone
    from mosd_hidden_dynamics.extended import load_extended_group, series_metrics
    from mosd_hidden_dynamics.neural_ode import advance, load_ode
    from mosd_hidden_dynamics.study import write_rows
    from mosd_hidden_dynamics.study_models import load_ridge, predict_ridge, rollout_ridge
    from mosd_hidden_dynamics.utils import save_json, sha256_file, source_record
    torch.set_num_threads(4)
    root = Path(args.run_dir).resolve()
    cfg = json.loads((root / "config_requested.json").read_text())
    base = json.loads((root / "backbone_config.json").read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    extra = json.loads((repo / "configs/remaining_hidden_dynamics.json").read_text())
    output = root / "ode_interpolation"
    output.mkdir(exist_ok=True)
    pipeline, _ = load_backbone(base, args.device)
    all_rows = []
    for dataset in cfg["datasets"]:
        if dataset["family"] == "control":
            continue
        name = dataset["name"]
        for protocol in ("in_series", "out_of_series"):
            directory = output / f"{name}_{protocol}"
            if (directory / "complete.json").exists():
                all_rows.extend(json.loads((directory / "metrics.json").read_text()))
                continue
            directory.mkdir(exist_ok=True)
            group, anchors, digest = load_extended_group(cfg, registry, root, name, protocol,
                                                           extra["history_order"], extra["ode"]["segment_steps"])
            truth, identifiers = [], []
            roundtrip, compared = 0, 0
            for record in registry:
                eligible = [a for a in anchors if a["split"] == "val" and a["series_id"] == record["series_id"]]
                if not eligible:
                    continue
                sources = np.asarray([a["source_end"] for a in eligible])
                endpoints = np.unique((sources[:, None] + np.arange(17)[None] * 4).reshape(-1))
                if endpoints.max() >= record["test_start"]:
                    raise ValueError("Dense interpolation attempted test extraction")
                path = directory / (record["series_id"] + "_dense.npz")
                if path.exists():
                    with np.load(path) as cached:
                        if not np.array_equal(cached["raw_endpoints"], endpoints):
                            raise ValueError("Dense endpoints changed")
                        states = cached["states"].astype(np.float32)
                else:
                    with np.load(root / "data" / (record["series_id"] + ".npz")) as data:
                        observed = data["observed"][:record["test_start"]]
                    states = extract_vectors(pipeline.model, observed, endpoints, [5], 512, cfg["batch_size"])[:, 0]
                    np.savez_compressed(path, raw_endpoints=endpoints, states=states)
                positions = np.searchsorted(endpoints, sources[:, None] + np.arange(17)[None] * 4)
                truth.append(states[positions])
                identifiers.extend([record["series_id"]] * len(sources))
                with np.load(root / "features/pretrained" / (record["series_id"] + ".npz")) as data:
                    coarse_ends, coarse = data["raw_endpoints"], data["states"][:, 0]
                selected = np.searchsorted(coarse_ends, sources)
                roundtrip += int(np.sum(states[positions[:, 0]].astype(np.float16) == coarse[selected]))
                compared += coarse[selected].size
            target = np.concatenate(truth)
            ids = np.asarray(identifiers)
            # load_extended_group concatenates records in this same registry order.
            if not np.array_equal(ids, group["val"]["ids"]):
                raise ValueError("Dense/coarse entity order mismatch")
            x = group["val"]["x"]
            baseline = root / "remaining_baselines" / f"{name}_{protocol}"
            if not (baseline / "complete.json").exists():
                raise ValueError("Baseline case must finish before interpolation")
            ridge1 = load_ridge(baseline / "ridge_direct_k1.npz", args.device)
            ridge4 = load_ridge(baseline / "ridge_direct_k4.npz", args.device)
            coarse_predictions = {0: x, **rollout_ridge(ridge1, x, [1, 2, 3, 4])}
            direct4 = predict_ridge(ridge4, x)
            bridges = []
            for j in range(17):
                lower, fraction = divmod(j, 4)
                fraction /= 4
                bridge = coarse_predictions[lower] if lower == 4 else ((1 - fraction) * coarse_predictions[lower]
                                                                 + fraction * coarse_predictions[lower + 1])
                bridges.append(bridge)
            linear = np.stack(bridges, axis=1)
            direct_bridge = x[:, None] + np.arange(17)[None, :, None] / 16 * (direct4 - x)[:, None]
            common = {"dataset": name, "protocol": protocol, "condition": "pretrained"}
            rows = []
            reference = group["train"]["x"].mean(0)
            def record(method, prediction, seed=None):
                for subset, indices in (("all_future", np.arange(1, 17)),
                                        ("interior", np.flatnonzero(np.arange(17) % 4)),
                                        ("endpoint_k4", np.array([16]))):
                    target_flat = target[:, indices].reshape(-1, target.shape[-1])
                    prediction_flat = prediction[:, indices].reshape(-1, target.shape[-1])
                    labels = np.repeat(ids, len(indices))
                    rows.extend({**common, "model": method, "subset": subset, "seed": seed, **r}
                                for r in series_metrics(target_flat, prediction_flat, reference, labels))
            record("ridge_recursive_linear_bridge", linear)
            record("ridge_direct_k4_linear_bridge", direct_bridge)
            record("persistence", np.broadcast_to(x[:, None], target.shape))
            hashes = {}
            completed = root / "neural_ode" / f"{name}_{protocol}" / "complete.json"
            waited = time.monotonic()
            while not completed.exists():
                finished = root / "neural_ode_exit.json"
                if finished.exists() or time.monotonic() - waited > 1800:
                    raise RuntimeError("ODE case did not complete; inspect its launcher log")
                time.sleep(2)
            for seed in extra["seeds"]:
                path = root / "neural_ode" / f"{name}_{protocol}" / f"ode_seed{seed}.pt"
                waited = time.monotonic()
                while not path.exists():
                    if time.monotonic() - waited > 1800:
                        raise TimeoutError("ODE fit did not arrive")
                    time.sleep(2)
                hashes[str(path.relative_to(root))] = sha256_file(path)
                fit = load_ode(path, args.device)
                with torch.inference_mode():
                    state = torch.as_tensor((x - fit["mean"]) / fit["scale"], dtype=torch.float32, device=args.device)
                    predictions = [x]
                    for _ in range(16):
                        state = advance(fit["model"], state, duration=.25)
                        predictions.append(state.cpu().numpy().astype(np.float64) * fit["scale"] + fit["mean"])
                predicted = np.stack(predictions, axis=1)
                record("neural_ode", predicted, seed)
                np.savez_compressed(directory / f"first_anchor_seed{seed}.npz", truth=target[0],
                                    ode=predicted[0], ridge=linear[0], direct_bridge=direct_bridge[0])
            save_json(directory / "metrics.json", rows)
            write_rows(directory / "metrics.csv", rows)
            save_json(directory / "run_metadata.json", {"source": source_record(repo), "anchor_sha256": digest,
                      "frozen_fits": hashes, "fractional_step_raw": 4, "training_step_raw": 16,
                      "coarse_state_fp16_value_match_fraction": roundtrip / compared,
                      "refitted": False, "test_evaluated": False,
                      "interpretation": "Forecasts of fresh held-out intermediate states from the initial state; no future endpoint conditioning."})
            save_json(directory / "complete.json", {"rows": len(rows), "test_evaluated": False})
            all_rows.extend(rows)
            print("DONE DENSE INTERPOLATION", name, protocol, flush=True)
            del group, truth, target
    save_json(output / "metrics.json", all_rows)
    write_rows(output / "metrics.csv", all_rows)
    save_json(output / "complete.json", {"groups": 6, "test_evaluated": False})


if __name__ == "__main__":
    main()
