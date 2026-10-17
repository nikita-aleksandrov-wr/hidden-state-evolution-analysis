import copy
import json
import time
from pathlib import Path

import numpy as np

from .data import generate_chaotic
from .extraction import extract_vectors, load_backbone, resolve_layers
from .utils import save_json, sha256_file


def electricity_rows(dataset, workspace, count):
    from huggingface_hub import hf_hub_download
    import pyarrow as pa

    path = hf_hub_download(dataset["repo_id"], dataset["filename"], repo_type="dataset",
                           revision=dataset["revision"], cache_dir=workspace / "cache/gift_eval")
    saved = 0
    with pa.memory_map(path, "r") as source:
        reader = pa.ipc.open_stream(source)
        for batch in reader:
            rows = batch.slice(0, min(len(batch), count - saved)).to_pylist()
            for row in rows:
                yield row
                saved += 1
            if saved >= count:
                break
    if saved < count:
        raise ValueError("The declared electricity subset is incomplete")


def causal_prefix(values, length):
    values = np.asarray(values, dtype=np.float32)
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) == 0:
        raise ValueError("No observed values")
    offset = int(finite[0])
    values = values[offset:offset + length].copy()
    if len(values) != length:
        raise ValueError("Series is shorter than the declared prefix")
    valid = np.isfinite(values)
    previous = np.maximum.accumulate(np.where(valid, np.arange(len(values)), 0))
    return values[previous], offset, int((~valid).sum())


def prepare_study_data(cfg, base_cfg, workspace, run_dir, logger):
    data_dir = run_dir / "data"
    data_dir.mkdir()
    registry = []
    for dataset_index, dataset in enumerate(cfg["datasets"]):
        count = dataset["n_train"] + dataset["n_val"]
        external = iter(electricity_rows(dataset, workspace, count)) if dataset["name"] == "electricity" else None
        for index in range(count + dataset["n_test"]):
            series_id = f"{dataset['name'].lower()}_{index:02d}"
            role = "train" if index < dataset["n_train"] else "val" if index < count else "test"
            record = {"series_id": series_id, "dataset": dataset["name"], "family": dataset["family"],
                      "role": role, "test_evaluated": False}
            if role == "test":
                record["status"] = "reserved_not_loaded"
                registry.append(record)
                continue
            seed = cfg["seed"] + 1000 * dataset_index + index
            if dataset["family"] == "chaotic":
                data_cfg = copy.deepcopy(base_cfg["data"])
                data_cfg.update(system=dataset["name"], trajectory_length=cfg["trajectory_length"], series_id=series_id)
                initial = np.asarray(dataset["initial_condition"])
                if index:
                    initial = initial + np.random.default_rng(seed).normal(size=3) * np.maximum(np.abs(initial), 1) * 0.1
                data_cfg["initial_condition"] = initial.tolist()
                observed, trajectory, times, metadata = generate_chaotic(data_cfg, seed)
                record["season_raw"] = 128
            elif dataset["name"] == "electricity":
                row = next(external)
                observed, offset, missing = causal_prefix(row["target"], cfg["trajectory_length"])
                trajectory = observed[:, None]
                times = np.arange(len(observed), dtype=np.float64) * 900
                metadata = {"repo_id": dataset["repo_id"], "revision": dataset["revision"],
                            "filename": dataset["filename"], "item_id": row["item_id"], "frequency": row["freq"],
                            "original_start": str(row["start"]), "leading_missing_trimmed": offset,
                            "missing_forward_filled": missing, "pretraining_overlap": "unknown"}
                record["season_raw"] = dataset["season_raw"]
                record["item_id"] = row["item_id"]
            else:
                rng = np.random.default_rng(seed)
                noise = rng.normal(size=cfg["trajectory_length"] + 1024)
                series = np.zeros_like(noise)
                for t in range(1, len(series)):
                    series[t] = dataset["phi"] * series[t - 1] + noise[t]
                observed = series[1024:].astype(np.float32)
                trajectory, times = observed[:, None], np.arange(len(observed), dtype=np.float64)
                metadata = {"phi": dataset["phi"], "seed": seed, "burn_in": 1024}
                record["season_raw"] = None
            if not np.isfinite(observed).all():
                raise ValueError("Non-finite observed prefix")
            path = data_dir / (series_id + ".npz")
            np.savez_compressed(path, observed=observed, trajectory=trajectory, times=times)
            record.update(length=len(observed), seed=seed, data_sha256=sha256_file(path),
                          test_start=int(len(observed) * 0.8), status="prepared", generation=metadata)
            registry.append(record)
            logger.info("Prepared %s (%s)", series_id, role)
    save_json(run_dir / "series_registry.json", registry)
    return registry


def extract_study_states(cfg, base_cfg, registry, run_dir, device, logger, selected_conditions=None):
    import torch
    from chronos.chronos2.model import Chronos2Model

    base_cfg = copy.deepcopy(base_cfg)
    base_cfg["model"]["batch_size"] = cfg["batch_size"]
    pipeline, seconds = load_backbone(base_cfg, device)
    layers = resolve_layers(base_cfg["model"], pipeline.model.config.num_layers)
    conditions = [("pretrained", None)] + [(f"random_{seed}", seed) for seed in cfg["random_init_seeds"]]
    if selected_conditions:
        conditions = [item for item in conditions if item[0] in selected_conditions]
        if len(conditions) != len(selected_conditions):
            raise ValueError("Unknown extraction condition")
    settings = {"context": cfg["context_length"], "model": base_cfg["model"],
                "source_sha256": {name: sha256_file(Path(__file__).parent / name)
                                  for name in ("study_data.py", "extraction.py")}}
    feature_dir = run_dir / "features"
    feature_dir.mkdir(exist_ok=True)
    for condition, seed in conditions:
        output_dir = feature_dir / condition
        output_dir.mkdir(exist_ok=True)
        if seed is None:
            model = pipeline.model
        else:
            torch.manual_seed(seed)
            model = Chronos2Model(copy.deepcopy(pipeline.model.config)).to(device=device, dtype=torch.float32).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        for record in registry:
            if record["role"] == "test":
                continue
            output = output_dir / (record["series_id"] + ".npz")
            metadata_path = output.with_suffix(".json")
            if output.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata["sha256"] != sha256_file(output) or metadata["data_sha256"] != record["data_sha256"]:
                    raise ValueError("Cached features changed")
                if metadata["settings"] != settings:
                    raise ValueError("Extraction settings/source changed; use a new study ID")
                continue
            start = time.perf_counter()
            with np.load(run_dir / "data" / (record["series_id"] + ".npz")) as data:
                observed = data["observed"]
            ends = np.arange(cfg["context_length"] - 1, record["test_start"], model.chronos_config.input_patch_stride)
            states = extract_vectors(model, observed, ends, layers, cfg["context_length"], cfg["batch_size"]).astype(np.float16)
            if not np.isfinite(states).all():
                raise ValueError("State storage overflow")
            np.savez_compressed(output, states=states, raw_endpoints=ends, layers=layers)
            torch.cuda.synchronize(model.device)
            save_json(metadata_path, {"condition": condition, "random_seed": seed, "layers": layers, "settings": settings,
                      "shape": list(states.shape), "seconds": time.perf_counter() - start,
                      "sha256": sha256_file(output), "data_sha256": record["data_sha256"],
                      "test_states_extracted": False, "future_values_supplied": False,
                      "inference_dtype": "float32", "storage_dtype": "float16"})
            logger.info("Extracted %s / %s: %d states in %.2f s", condition, record["series_id"], len(ends), time.perf_counter() - start)
        if seed is not None:
            del model
            torch.cuda.empty_cache()
    save_json(run_dir / "extraction.json", {"model": base_cfg["model"], "conditions": [c[0] for c in conditions],
              "layers": layers, "model_load_seconds": seconds, "test_states_extracted": False})


def study_pairs(ends, record, context, max_horizon, protocol, stride=16):
    if record["role"] == "test":
        raise ValueError("Test series are reserved")
    if protocol == "in_series":
        boundary = int(record["length"] * 0.7)
        intervals = {"train": (0, boundary), "val": (boundary + context, record["test_start"])}
    elif protocol == "out_of_series":
        intervals = {record["role"]: (0, record["test_start"])}
    else:
        raise ValueError("Unknown study protocol")
    result = {}
    for split, (start, stop) in intervals.items():
        indices = np.flatnonzero((ends - stride - context + 1 >= start) & (ends + max_horizon * stride < stop))
        if len(indices) == 0:
            raise ValueError("No eligible anchors")
        if not np.all(np.diff(ends) == stride):
            raise ValueError("Irregular state endpoints")
        result[split] = indices
    return result
