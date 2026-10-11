import time

import numpy as np

from .utils import save_json, sha256_file


def resolve_layers(model_cfg, num_layers):
    middle = model_cfg["middle_layer_index"]
    if middle is None:
        middle = num_layers // 2 - 1
    layers = [middle]
    if model_cfg["capture_next_layer"]:
        layers.append(middle + 1)
    if any(i < 0 or i >= num_layers for i in layers):
        raise ValueError("Requested block indices are outside the loaded encoder")
    return layers


def load_backbone(cfg, device):
    import torch
    from chronos import Chronos2Pipeline

    if cfg["model"]["dtype"] != "float32":
        raise ValueError("The first checkpoint uses float32 inference")
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    pipeline = Chronos2Pipeline.from_pretrained(
        cfg["model"]["id"], revision=cfg["model"]["revision"],
        device_map=device, torch_dtype=torch.float32, attn_implementation="eager",
    )
    model = pipeline.model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "block"):
        raise RuntimeError("Installed Chronos-2 does not expose the expected encoder blocks")
    if device.startswith("cuda"):
        torch.cuda.synchronize(model.device)
    return pipeline, time.perf_counter() - start


def extract_vectors(model, observed, endpoints, layers, context, batch_size):
    import torch

    patch = model.chronos_config.input_patch_size
    stride = model.chronos_config.input_patch_stride
    if patch != stride or context % stride:
        raise ValueError("This extraction protocol requires non-overlapping, complete context patches")
    num_patches = context // stride
    if context > model.chronos_config.context_length:
        raise ValueError("Requested context exceeds the loaded model limit")
    selected_index = num_patches - 1
    expected_tokens = num_patches + int(model.chronos_config.use_reg_token) + 1
    captured = {}

    def make_hook(layer):
        def hook(module, inputs, output):
            hidden = output.hidden_states
            if hidden.ndim != 3 or hidden.shape[1:] != (expected_tokens, model.config.d_model):
                raise RuntimeError(f"Unexpected block output shape: {tuple(hidden.shape)}")
            captured[layer] = hidden[:, selected_index].detach().cpu().numpy().copy()
        return hook

    hooks = [model.encoder.block[i].register_forward_hook(make_hook(i)) for i in layers]
    batches = []
    try:
        with torch.inference_mode():
            for start in range(0, len(endpoints), batch_size):
                batch_ends = endpoints[start:start + batch_size]
                windows = np.stack([observed[e - context + 1:e + 1] for e in batch_ends])
                if windows.shape[1] != context or not np.isfinite(windows).all():
                    raise ValueError("Invalid rolling contexts")
                batch = torch.as_tensor(windows, dtype=torch.float32, device=model.device)
                # Each rolling window is an independent task: no group attention between windows.
                groups = torch.arange(len(batch), dtype=torch.long, device=model.device)
                captured.clear()
                outputs = model.encode(context=batch, group_ids=groups, num_output_patches=1)
                if outputs[3] != num_patches or set(captured) != set(layers):
                    raise RuntimeError("Context-patch count or captured layers do not match")
                batches.append(np.stack([captured[layer] for layer in layers], axis=1))
    finally:
        for hook in hooks:
            hook.remove()
    result = np.concatenate(batches)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite hidden states")
    return result


def extract_states(cfg, observed, rows, run_dir, device, logger):
    import torch

    pipeline, load_seconds = load_backbone(cfg, device)
    model = pipeline.model
    layers = resolve_layers(cfg["model"], model.config.num_layers)
    context = cfg["state_extraction"]["context_length_raw"]
    endpoints = sorted({
        r[key] for r in rows if r["eligible"] and r["split"] in ("train", "val")
        for key in ("raw_context_end", "target_context_end")
    })
    logger.info("Extracting %d states at blocks %s; terminal observed patch only", len(endpoints), layers)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)
    start = time.perf_counter()
    states = extract_vectors(model, observed, endpoints, layers, context, cfg["model"]["batch_size"])
    if device.startswith("cuda"):
        torch.cuda.synchronize(model.device)
    extraction_seconds = time.perf_counter() - start
    storage = cfg["state_extraction"]["storage_dtype"]
    if storage not in ("float16", "float32"):
        raise ValueError("Unsupported activation storage dtype")
    stored = states.astype(storage)
    if not np.isfinite(stored).all():
        raise ValueError("Hidden states overflow the selected storage dtype")
    path = run_dir / "states.npz"
    np.savez_compressed(path, states=stored, raw_endpoints=endpoints, layers=layers)
    metadata = {
        "model_id": cfg["model"]["id"], "model_revision": cfg["model"]["revision"],
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "resolved_model_config": model.config.to_dict(),
        "layers_zero_based": layers, "layer_position": "post_block_pre_final_layernorm",
        "context_length_raw": context, "patch_size": model.chronos_config.input_patch_size,
        "patch_stride": model.chronos_config.input_patch_stride,
        "num_context_patches": context // model.chronos_config.input_patch_stride,
        "selected_context_patch_index": context // model.chronos_config.input_patch_stride - 1,
        "reg_token_present": model.chronos_config.use_reg_token,
        "masked_output_patches": 1, "future_targets_or_covariates_supplied": False,
        "group_policy": "one independent group per rolling window",
        "native_window_normalization": "Chronos-2 InstanceNorm with arcsinh from past context only",
        "shape": list(states.shape), "inference_dtype": "float32", "storage_dtype": storage,
        "splits_extracted": ["train", "val"], "test_states_extracted": False,
        "file_sha256": sha256_file(path), "device": str(model.device),
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "all_parameters_frozen": all(not p.requires_grad for p in model.parameters()),
    }
    if device.startswith("cuda"):
        properties = torch.cuda.get_device_properties(model.device)
        metadata["gpu"] = {"name": properties.name, "total_memory_bytes": properties.total_memory,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(model.device),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(model.device)}
    save_json(run_dir / "extraction_metadata.json", metadata)
    return metadata, {"model_load_seconds": load_seconds, "extraction_seconds": extraction_seconds}
