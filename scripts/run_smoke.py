import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare data, extract frozen states, and evaluate validation baselines")
    parser.add_argument("--config", required=True, help="JSON experiment configuration")
    parser.add_argument("--workspace", default=None, help="Storage directory outside the source repository")
    parser.add_argument("--run-id", required=True, help="New run directory name")
    parser.add_argument("--device", default="cuda:0", help="Explicit inference device")
    parser.add_argument("--stage", choices=("all", "prepare", "extract", "baseline"), default="all")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(cfg["threads"])
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mosd_hidden_dynamics.pipeline import run_experiment

    run_experiment(cfg, args.workspace or repo.parent, args.run_id, args.stage, args.device, repo)


if __name__ == "__main__":
    main()
