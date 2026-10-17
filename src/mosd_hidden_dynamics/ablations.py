import hashlib
import json
from collections import defaultdict

import numpy as np

from .extraction import extract_vectors
from .study_data import study_pairs
from .utils import save_json, sha256_file


def extract_variant(model, observed, endpoints, context, batch_size, layers, shuffle_seed=None):
    import torch
    hook = None
    if shuffle_seed is not None:
        def positions(module, args, kwargs):
            length = kwargs["inputs_embeds"].shape[1]
            patches = context // model.chronos_config.input_patch_stride
            order = np.arange(length)
            order[:patches] = np.random.default_rng(shuffle_seed).permutation(patches)
            kwargs = dict(kwargs)
            kwargs["position_ids"] = torch.as_tensor(order, dtype=torch.long, device=model.device)[None]
            return args, kwargs
        hook = model.encoder.register_forward_pre_hook(positions, with_kwargs=True)
    try:
        states = extract_vectors(model, observed, endpoints, layers, context, batch_size)
    finally:
        if hook is not None:
            hook.remove()
    with torch.inference_mode():
        last = torch.as_tensor(states[:, -1], device=model.device, dtype=torch.float32)
        normalized = model.encoder.final_layer_norm(last).cpu().numpy()
    return np.concatenate([states, normalized[:, None]], axis=1)


def training_raw_statistics(cfg, registry, root):
    result = {}
    for dataset in cfg["datasets"]:
        series = []
        for record in registry:
            if record["dataset"] == dataset["name"] and record["role"] == "train":
                with np.load(root / "data" / (record["series_id"] + ".npz")) as data:
                    series.append(data["observed"][:int(record["length"] * .7)].astype(np.float64))
        values = np.concatenate(series)
        mean, variance = float(values.mean()), float(values.var())
        numerator = sum(float(((s[:-1] - mean) * (s[1:] - mean)).sum()) for s in series)
        denominator = sum(float(np.square(s[:-1] - mean).sum()) for s in series)
        phi = float(np.clip(numerator / denominator, -.99, .99))
        result[dataset["name"]] = {"mean": mean, "std": np.sqrt(variance), "phi": phi,
                                   "fit": "training identities, first 70% only; no cross-series lag pairs"}
    return result


def variant_specs(extra):
    specs = [{"name": f"context_{c}", "context": c} for c in extra["ablations"]["contexts"]]
    specs.append({"name": "shuffled_positions", "context": 512,
                  "shuffle_seed": extra["ablations"]["position_shuffle_seed"]})
    specs.extend({"name": f"noise_{level}", "context": 512, "noise": level}
                 for level in extra["ablations"]["observation_noise_std"])
    specs.append({"name": "matched_ar1", "context": 512, "matched_ar1": True})
    return specs


def prepare_variant_observations(spec, record, original, stats, seed):
    rng = np.random.default_rng(seed + record["seed"])
    if spec.get("matched_ar1"):
        noise = rng.normal(size=len(original) + 1024) * stats["std"] * np.sqrt(1 - stats["phi"] ** 2)
        values = np.zeros_like(noise)
        for index in range(1, len(values)):
            values[index] = stats["phi"] * values[index - 1] + noise[index]
        return (values[1024:] + stats["mean"]).astype(np.float32)
    if spec.get("noise"):
        return (original + rng.normal(size=len(original)) * spec["noise"] * stats["std"]).astype(np.float32)
    return original


def extract_ablations(cfg, base, extra, registry, root, device, selected=None):
    from .extraction import load_backbone
    pipeline, _ = load_backbone(base, device)
    model = pipeline.model
    layers = extra["ablations"]["layers"]
    if layers[-1] != model.config.num_layers - 1:
        raise ValueError("Native final-norm comparison must use the last block")
    output = root / "ablations"
    output.mkdir(exist_ok=True)
    stats = training_raw_statistics(cfg, registry, root)
    settings = {"model": base["model"], "layers": layers, "source_sha256": sha256_file(__file__)}
    save_json(output / "training_raw_statistics.json", stats)
    for spec in variant_specs(extra):
        if selected and spec["name"] not in selected:
            continue
        directory = output / "features" / spec["name"]
        directory.mkdir(parents=True, exist_ok=True)
        for record in registry:
            if record["role"] == "test":
                continue
            family = next(d["family"] for d in cfg["datasets"] if d["name"] == record["dataset"])
            if spec.get("noise") and family != "chaotic" or spec.get("matched_ar1") and family == "control":
                continue
            path = directory / (record["series_id"] + ".npz")
            if path.exists():
                metadata = json.loads(path.with_suffix(".json").read_text())
                if (sha256_file(path) != metadata["sha256"] or metadata["spec"] != spec
                        or metadata["settings"] != settings):
                    raise ValueError("Cached ablation changed")
                continue
            raw_path = root / "data" / (record["series_id"] + ".npz")
            if sha256_file(raw_path) != record["data_sha256"]:
                raise ValueError("Raw data changed")
            with np.load(raw_path) as data:
                original = data["observed"][:record["test_start"]]
            observed = prepare_variant_observations(spec, record, original, stats[record["dataset"]], extra["seed"])
            endpoints = np.arange(spec["context"] - 1, record["test_start"], 16)
            states = extract_variant(model, observed, endpoints, spec["context"], cfg["batch_size"],
                                     layers, spec.get("shuffle_seed")).astype(np.float16)
            if not np.isfinite(states).all():
                raise ValueError("Ablation state overflow")
            np.savez_compressed(path, states=states, raw_endpoints=endpoints, observed=observed)
            save_json(path.with_suffix(".json"), {"spec": spec, "settings": settings, "sha256": sha256_file(path),
                      "data_sha256": record["data_sha256"], "labels": [f"block{i}" for i in layers] + ["final_norm"],
                      "test_states_extracted": False, "future_values_supplied": False,
                      "position_control": "context position IDs permuted; REG/output IDs fixed; rotary positions, not learned embeddings"})
            print("EXTRACTED", spec["name"], record["series_id"], len(endpoints), flush=True)
    save_json(output / "extraction_complete.json", {"variants": selected or [s["name"] for s in variant_specs(extra)],
              "test_evaluated": False})


def load_ablation_group(cfg, registry, root, dataset, protocol, variant, label, horizons=(1, 4),
                        stride=16, calibrate=False, calibration=128, common_context=1024):
    chunks = {name: defaultdict(list) for name in ("train", "val")}
    manifest = []
    for record in registry:
        if record["dataset"] != dataset or record["role"] == "test":
            continue
        path = root / "ablations/features" / variant / (record["series_id"] + ".npz")
        metadata = json.loads(path.with_suffix(".json").read_text())
        if (sha256_file(path) != metadata["sha256"] or metadata["test_states_extracted"]
                or metadata["data_sha256"] != record["data_sha256"]):
            raise ValueError("Ablation cache integrity failed")
        context = metadata["spec"]["context"]
        with np.load(path) as data:
            ends = data["raw_endpoints"][::stride // 16]
            states = data["states"][::stride // 16, metadata["labels"].index(label)].astype(np.float64)
            observed = data["observed"]
        if np.any(ends >= record["test_start"]):
            raise ValueError("Test data were encoded")
        mean, scale = states[:calibration].mean(0), states[:calibration].std(0)
        scale = np.where(scale > 1e-8, scale, 1)
        comparison_context = max(context, common_context)
        for split, indices in study_pairs(ends, record, comparison_context, max(horizons), protocol, stride=stride).items():
            if calibrate:
                indices = indices[indices >= calibration]
            if not len(indices):
                raise ValueError("Empty ablation anchors")
            part = chunks[split]
            part["x"].append(states[indices])
            part["z"].append((states[indices] - mean) / scale)
            part["calibration_mean"].append(np.broadcast_to(mean, (len(indices), len(mean))))
            part["calibration_scale"].append(np.broadcast_to(scale, (len(indices), len(scale))))
            part["raw"].append(np.stack([observed[e - context + 1:e + 1] for e in ends[indices]]))
            part["ids"].append(np.repeat(record["series_id"], len(indices)))
            for k in horizons:
                part[f"y{k}"].append(states[indices + k])
                part[f"zy{k}"].append((states[indices + k] - mean) / scale)
            manifest.extend({"series_id": record["series_id"], "split": split, "source_end": int(ends[i]),
                             "receptive_start": int(ends[i] - stride - comparison_context + 1),
                             "max_target_end": int(ends[i] + max(horizons) * stride)} for i in indices)
    group = {name: {key: np.concatenate(values) for key, values in part.items()} for name, part in chunks.items()}
    if protocol == "out_of_series" and set(group["train"]["ids"]) & set(group["val"]["ids"]):
        raise ValueError("Ablation entity overlap")
    for series in {row["series_id"] for row in manifest}:
        a = [row for row in manifest if row["series_id"] == series and row["split"] == "train"]
        b = [row for row in manifest if row["series_id"] == series and row["split"] == "val"]
        if a and b and min(row["receptive_start"] for row in b) - max(row["max_target_end"] for row in a) - 1 < context:
            raise ValueError("Ablation embargo failed")
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return group, manifest, digest
