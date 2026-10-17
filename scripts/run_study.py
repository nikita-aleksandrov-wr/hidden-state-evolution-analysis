import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Multi-series, multi-horizon hidden-state experiments")
    parser.add_argument("--config", default="configs/study_hidden_dynamics.json")
    parser.add_argument("--workspace", default="..")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("prepare", "extract", "evaluate"), required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--evaluation-id", default="main")
    parser.add_argument("--skip-mlp", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(cfg["threads"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    repo = Path(__file__).resolve().parents[1]
    workspace = Path(args.workspace).resolve()
    if workspace.is_relative_to(repo):
        raise ValueError("Workspace must stay outside the source repository")
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.run_id):
        raise ValueError("Invalid run ID")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.evaluation_id):
        raise ValueError("Invalid evaluation ID")
    os.environ["HF_HOME"] = str(workspace / "cache/huggingface")
    sys.path.insert(0, str(repo / "src"))
    import torch
    from mosd_hidden_dynamics.study_data import extract_study_states, prepare_study_data
    from mosd_hidden_dynamics.utils import environment_record, save_json, source_record
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base_cfg = json.loads((repo / cfg["baseline_config"]).read_text())
    run_dir = workspace / "artifacts" / args.run_id
    if args.stage == "prepare":
        run_dir.mkdir(parents=True, exist_ok=False)
        save_json(run_dir / "config_requested.json", cfg)
        save_json(run_dir / "backbone_config.json", base_cfg)
    elif json.loads((run_dir / "config_requested.json").read_text()) != cfg:
        raise ValueError("Study config changed; use a new study ID")
    elif json.loads((run_dir / "backbone_config.json").read_text()) != base_cfg:
        raise ValueError("Backbone config changed; use a new study ID")
    logger = logging.getLogger("mosd.study")
    logger.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(), logging.FileHandler(run_dir / f"{args.stage}.log")):
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    start = time.perf_counter()
    metadata = {"stage": args.stage, "source": source_record(repo), "environment": environment_record(),
                "test_evaluated": False, "status": "running", "device": args.device}
    try:
        if args.stage == "prepare":
            prepare_study_data(cfg, base_cfg, workspace, run_dir, logger)
        else:
            registry = json.loads((run_dir / "series_registry.json").read_text())
            if args.stage == "extract":
                extract_study_states(cfg, base_cfg, registry, run_dir, args.device, logger, args.conditions)
            else:
                from mosd_hidden_dynamics.study import evaluate_study
                evaluate_study(cfg, base_cfg, registry, run_dir, repo, args.device, logger,
                               args.evaluation_id, args.conditions, args.datasets, args.skip_mlp)
        metadata["status"] = "complete"
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        logger.exception("Stage failed")
        raise
    finally:
        metadata["seconds"] = time.perf_counter() - start
        save_json(run_dir / f"{args.stage}_metadata.json", metadata)


if __name__ == "__main__":
    main()
