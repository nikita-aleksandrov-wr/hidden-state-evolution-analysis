import csv
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np

from .baselines import fit_baselines
from .data import prepare_data
from .extraction import extract_states
from .splits import audit_manifest, load_manifest, make_manifest, save_manifest
from .utils import environment_record, save_json, sha256_file, source_record


def resolve_model_config(cfg):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=cfg["model"]["id"], revision=cfg["model"]["revision"], filename="config.json"
    )
    model_config = json.loads(Path(path).read_text(encoding="utf-8"))
    if model_config["chronos_config"]["input_patch_size"] != model_config["chronos_config"]["input_patch_stride"]:
        raise ValueError("The checkpoint protocol requires equal patch size and stride")
    return model_config


def write_summary(cfg, metrics, extraction, run_dir):
    persistence = metrics["baselines"]["persistence"]
    rows = []
    for name, values in metrics["baselines"].items():
        rows.append({
            "dataset": "Lorenz coordinate 0", "model": cfg["model"]["id"],
            "layer_zero_based": extraction["layers_zero_based"][0],
            "context_raw": cfg["state_extraction"]["context_length_raw"],
            "horizon_patches": cfg["prediction"]["horizon_patches"], "seed": cfg["seed"],
            "split": "validation", "baseline": name,
            "r2": f"{values['r2']:.3f}", "cosine": f"{values['cosine']:.3f}",
            "delta_r2_over_persistence": f"{values['r2'] - persistence['r2']:.3f}",
            "delta_cosine_over_persistence": f"{values['cosine'] - persistence['cosine']:.3f}",
        })
    with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(run_dir, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(metrics["baselines"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, metric in zip(axes, ("r2", "cosine")):
        values = [metrics["baselines"][name][metric] for name in names]
        ax.bar(range(len(names)), values, color=["#999999", "#557a95", "#d36b35", "#b4a7bd"])
        ax.set_xticks(range(len(names)), ["Train mean", "Persistence", "Ridge", "Shuffled ridge"][:len(names)], rotation=20)
        ax.set_ylabel("R² (training-mean reference)" if metric == "r2" else "Mean cosine similarity")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Lorenz / Chronos-2: preliminary validation results")
    fig.tight_layout()
    fig.savefig(run_dir / "baseline_validation.png", dpi=180)
    plt.close(fig)
    with np.load(run_dir / "data.npz") as data:
        trajectory, times = data["trajectory"], data["times"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(times[:1024], trajectory[:1024, 0], linewidth=0.8)
    axes[0].set(xlabel="Physical system time", ylabel="Observed Lorenz x")
    axes[1].plot(trajectory[:4096, 0], trajectory[:4096, 2], linewidth=0.4)
    axes[1].set(xlabel="Lorenz x", ylabel="Lorenz z")
    fig.suptitle("Generated deterministic Lorenz trajectory; no observation noise")
    fig.tight_layout()
    fig.savefig(run_dir / "lorenz_data.png", dpi=180)
    plt.close(fig)


def run_experiment(cfg, workspace, run_id, stage, device, repo):
    workspace, repo = Path(workspace).resolve(), Path(repo).resolve()
    if workspace == repo or workspace.is_relative_to(repo):
        raise ValueError("Workspace/data/cache/artifacts must be outside the source repository")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", run_id):
        raise ValueError("Use a simple run ID of up to 64 characters")
    if cfg["state_extraction"]["step_in_patches"] != 1:
        raise ValueError("The first checkpoint advances by exactly one model patch stride")
    os.environ["HF_HOME"] = str(workspace / "cache" / "huggingface")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run_dir = workspace / "artifacts" / run_id
    if stage in ("all", "prepare"):
        run_dir.mkdir(parents=True, exist_ok=False)
        save_json(run_dir / "config_requested.json", cfg)
        metadata = {"run_id": run_id, "environment": environment_record(),
                    "source": source_record(repo), "seed": cfg["seed"],
                    "test_evaluated": False, "status": "running"}
        runtime = {}
    else:
        requested = json.loads((run_dir / "config_requested.json").read_text())
        if requested != cfg:
            raise ValueError("Config differs from the prepared run; create a new run ID")
        metadata = json.loads((run_dir / "run_metadata.json").read_text())
        if metadata["source"]["sha256"] != source_record(repo)["sha256"]:
            raise ValueError("Source changed since preparation; create a new run ID")
        runtime = json.loads((run_dir / "runtime.json").read_text())
    logger = logging.getLogger("mosd")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in (logging.StreamHandler(), logging.FileHandler(run_dir / "run.log", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    start = time.perf_counter()
    logger.info("Run %s; stage=%s; device=%s; seed=%s", run_id, stage, device, cfg["seed"])
    try:
        if stage in ("all", "prepare"):
            fetch_start = time.perf_counter()
            model_config = resolve_model_config(cfg)
            save_json(run_dir / "model_config.json", model_config)
            runtime["model_config_fetch_seconds"] = time.perf_counter() - fetch_start
            observed, data_metadata, seconds = prepare_data(cfg, run_dir)
            runtime["data_preparation_seconds"] = seconds
            rows, bounds = make_manifest(
                len(observed), cfg["state_extraction"]["context_length_raw"],
                model_config["chronos_config"]["input_patch_stride"],
                cfg["prediction"]["horizon_patches"], **cfg["split"],
            )
            audit = audit_manifest(rows, bounds, observed)
            save_manifest(run_dir / "split_manifest.csv", rows)
            save_json(run_dir / "split_audit.json", audit)
            metadata["data"] = data_metadata
            metadata["split_manifest_sha256"] = sha256_file(run_dir / "split_manifest.csv")
            metadata["split_audit"] = audit
            logger.info("Data prepared; eligible pairs: %s; embargo audit passed", audit["eligible_pairs"])
        else:
            if sha256_file(run_dir / "data.npz") != metadata["data"]["file_sha256"]:
                raise ValueError("Prepared data checksum changed")
            if sha256_file(run_dir / "split_manifest.csv") != metadata["split_manifest_sha256"]:
                raise ValueError("Split manifest checksum changed")
            with np.load(run_dir / "data.npz") as data:
                observed = data["observed"]
            rows = load_manifest(run_dir / "split_manifest.csv")
        if stage in ("all", "extract"):
            if (run_dir / "states.npz").exists():
                raise FileExistsError("States already exist; use a new run ID")
            extraction, timings = extract_states(cfg, observed, rows, run_dir, device, logger)
            metadata["extraction"] = extraction
            runtime.update(timings)
        if stage in ("all", "baseline"):
            if (run_dir / "metrics.json").exists():
                raise FileExistsError("Metrics already exist; use a new run ID")
            if stage == "baseline":
                extraction = json.loads((run_dir / "extraction_metadata.json").read_text())
                if sha256_file(run_dir / "states.npz") != extraction["file_sha256"]:
                    raise ValueError("Activation checksum changed")
            metrics, timings = fit_baselines(cfg, rows, run_dir)
            metrics["split_manifest_sha256"] = metadata["split_manifest_sha256"]
            save_json(run_dir / "metrics.json", metrics)
            runtime.update(timings)
            write_summary(cfg, metrics, extraction, run_dir)
            plot_results(run_dir, metrics)
            metadata["status"] = "complete"
            for name, values in metrics["baselines"].items():
                logger.info("Validation %s: R2=%.3f, cosine=%.3f", name, values["r2"], values["cosine"])
        else:
            metadata["status"] = f"{stage}_complete"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("Run failed")
        raise
    finally:
        runtime[f"{stage}_wall_seconds"] = time.perf_counter() - start
        runtime["total_wall_seconds"] = sum(runtime.get(f"{s}_wall_seconds", 0) for s in ("all", "prepare", "extract", "baseline"))
        metadata["device_requested"] = device
        save_json(run_dir / "runtime.json", runtime)
        save_json(run_dir / "run_metadata.json", metadata)
    logger.info("Outputs: %s", run_dir)
    return run_dir
