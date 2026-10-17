import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Additional chaotic systems, seasonal domain and larger entity holdouts")
    parser.add_argument("--workspace", default="..")
    parser.add_argument("--run-id", default="additional_20261017")
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--stage", choices=("prepare", "extract", "evaluate", "transfer"), required=True)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mosd_hidden_dynamics.study import evaluate_study, load_group, write_rows
    from mosd_hidden_dynamics.study_data import causal_prefix, extract_study_states
    from mosd_hidden_dynamics.study_models import load_ridge, predict_ridge, rollout_ridge
    from mosd_hidden_dynamics.extended import series_metrics
    from mosd_hidden_dynamics.utils import environment_record, save_json, sha256_file, source_record
    import logging
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    workspace = Path(args.workspace).resolve()
    if workspace.is_relative_to(repo) or not args.run_id.replace("_", "").isalnum():
        raise ValueError("Use a valid run ID and an external workspace")
    root = workspace / "artifacts" / args.run_id
    reference = Path(args.reference_run).resolve()
    base = json.loads((reference / "backbone_config.json").read_text())
    cfg = json.loads((reference / "config_requested.json").read_text())
    cfg["random_init_seeds"] = []
    revision = next(d["revision"] for d in cfg["datasets"] if d["name"] == "electricity")
    cfg["datasets"] = [
        {"name": "Thomas", "family": "chaotic", "n_train": 8, "n_val": 8, "n_test": 4},
        {"name": "Aizawa", "family": "chaotic", "n_train": 8, "n_val": 8, "n_test": 4},
        {"name": "solar", "family": "seasonal", "n_train": 8, "n_val": 8, "n_test": 4,
         "repo_id": "Salesforce/GiftEval", "revision": revision, "filename": "solar/10T/data-00000-of-00001.arrow",
         "offset": 0, "season_raw": 144},
        {"name": "electricity_extra", "family": "seasonal", "n_train": 8, "n_val": 8, "n_test": 4,
         "repo_id": "Salesforce/GiftEval", "revision": revision, "filename": "electricity/15T/data-00000-of-00001.arrow",
         "offset": 16, "season_raw": 96}
    ]
    logger = logging.getLogger("mosd.additional")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    started = time.perf_counter()
    if args.stage == "prepare":
        root.mkdir(parents=True, exist_ok=False)
        (root / "data").mkdir()
        os.environ["HF_HOME"] = str(workspace / "cache/huggingface")
        registry = []
        from dysts import flows
        for di, dataset in enumerate(cfg["datasets"]):
            rows = None
            if dataset["family"] == "seasonal":
                from huggingface_hub import hf_hub_download
                import pyarrow as pa
                downloaded = hf_hub_download(dataset["repo_id"], dataset["filename"], repo_type="dataset",
                                            revision=dataset["revision"], cache_dir=workspace / "cache/gift_eval")
                rows, offset, remaining = [], dataset["offset"], dataset["n_train"] + dataset["n_val"]
                with pa.memory_map(downloaded, "r") as source:
                    for batch in pa.ipc.open_stream(source):
                        skip = min(offset, len(batch))
                        offset -= skip
                        take = min(remaining, len(batch) - skip)
                        if take:
                            rows.extend(batch.slice(skip, take).to_pylist())
                            remaining -= take
                        if remaining == 0:
                            break
                if remaining:
                    raise ValueError("Source lacks the declared additional entities")
            for index in range(dataset["n_train"] + dataset["n_val"] + dataset["n_test"]):
                role = "train" if index < dataset["n_train"] else "val" if index < dataset["n_train"] + dataset["n_val"] else "test"
                sid = dataset["name"].lower() + f"_{index:02d}"
                record = {"dataset": dataset["name"], "family": dataset["family"], "series_id": sid, "role": role,
                          "test_evaluated": False}
                if role == "test":
                    record["status"] = "reserved_not_loaded"
                    registry.append(record)
                    continue
                seed = 12026 + 1000 * di + index
                if dataset["family"] == "chaotic":
                    model = getattr(flows, dataset["name"])()
                    initial = np.asarray(model.ic, dtype=np.float64)
                    if index:
                        initial = initial + np.random.default_rng(seed).normal(size=len(initial)) * np.maximum(np.abs(initial), 1) * .1
                    model.ic = initial.copy()
                    settings = base["data"]
                    n = cfg["trajectory_length"] + settings["burn_in"]
                    times, values = model.make_trajectory(n, resample=True, pts_per_period=settings["points_per_period"],
                        return_times=True, standardize=False, postprocess=False, noise=0., method="DOP853",
                        random_seed=seed, rtol=settings["rtol"], atol=settings["atol"])
                    if values is None or values.shape != (n, 3):
                        raise ValueError("Additional system integration failed")
                    trajectory = np.asarray(values[settings["burn_in"]:], dtype=np.float64)
                    times = np.asarray(times[settings["burn_in"]:], dtype=np.float64)
                    observed = trajectory[:, 0].astype(np.float32)
                    metadata = {"system": dataset["name"], "initial_condition": initial.tolist(),
                                "parameters": {k: float(v) for k, v in model.params.items()}, "seed": seed,
                                "sampling_dt": float(times[1] - times[0]), "method": "DOP853", "noise": 0,
                                "standardize": False, "postprocess": False, "generation": settings}
                    record["season_raw"] = settings["points_per_period"]
                else:
                    row = rows[index]
                    observed, trim, missing = causal_prefix(row["target"], cfg["trajectory_length"])
                    trajectory, times = observed[:, None], np.arange(len(observed), dtype=np.float64)
                    metadata = {"repo_id": dataset["repo_id"], "revision": dataset["revision"], "filename": dataset["filename"],
                                "item_id": row["item_id"], "freq": row["freq"], "leading_missing_trimmed": trim,
                                "missing_forward_filled": missing, "source_entity_index": dataset["offset"] + index,
                                "pretraining_overlap": "unknown"}
                    record.update(item_id=row["item_id"], season_raw=dataset["season_raw"])
                if len(observed) != cfg["trajectory_length"] or not np.isfinite(observed).all():
                    raise ValueError("Invalid additional series")
                path = root / "data" / (sid + ".npz")
                np.savez_compressed(path, observed=observed, trajectory=trajectory, times=times)
                record.update(length=len(observed), seed=seed, data_sha256=sha256_file(path),
                              test_start=int(.8 * len(observed)), status="prepared", generation=metadata)
                registry.append(record)
                print("PREPARED", sid, role, flush=True)
        save_json(root / "config_requested.json", cfg)
        save_json(root / "backbone_config.json", base)
        save_json(root / "series_registry.json", registry)
    else:
        if json.loads((root / "config_requested.json").read_text()) != cfg:
            raise ValueError("Additional-study config changed")
        registry = json.loads((root / "series_registry.json").read_text())
        os.environ["HF_HOME"] = str(workspace / "cache/huggingface")
        if args.stage == "extract":
            extract_study_states(cfg, base, registry, root, args.device, logger)
        elif args.stage == "evaluate":
            evaluate_study(cfg, base, registry, root, repo, args.device, logger, "main")
        else:
            output = root / "frozen_transfer"
            output.mkdir(exist_ok=False)
            rows = []
            hashes = {}
            for dataset in cfg["datasets"]:
                group, anchors = load_group(cfg, registry, root, dataset["name"], "pretrained", "out_of_series")
                train, val = group["train"], group["val"]
                write_rows(output / f"{dataset['name']}_anchors.csv", anchors)
                for source in ("Lorenz", "Rossler", "electricity"):
                    for k in (1, 4, 64):
                        path = reference / "evaluations/main" / f"pretrained_{source}_out_of_series" / f"ridge_k{k}.npz"
                        hashes[str(path)] = sha256_file(path)
                        prediction = predict_ridge(load_ridge(path, args.device), val["x"])
                        rows.extend({"source_dataset": source, "dataset": dataset["name"], "model": "ridge_direct",
                                     "horizon": k, "protocol": "out_of_series", **r} for r in
                                    series_metrics(val[f"y{k}"], prediction, train[f"y{k}"].mean(0), val["ids"]))
                del group, train, val
            save_json(output / "metrics.json", rows)
            write_rows(output / "metrics.csv", rows)
            save_json(output / "run_metadata.json", {"source": source_record(repo), "frozen_fit_files": hashes,
                      "refitted": False, "target_training_mean_metric_reference_only": True, "test_evaluated": False})
            save_json(output / "complete.json", {"rows": len(rows), "test_evaluated": False})
    save_json(root / f"{args.stage}_metadata.json", {"source": source_record(repo), "seconds": time.perf_counter() - started,
              "environment": environment_record(), "status": "complete", "test_evaluated": False, "device": args.device})


if __name__ == "__main__":
    main()
