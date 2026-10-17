import argparse
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Matched temporal/depth MLPs on AR1 and white-noise controls")
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
    from mosd_hidden_dynamics.extended import load_extended_group, series_metrics
    from mosd_hidden_dynamics.study import write_rows
    from mosd_hidden_dynamics.study_models import fit_mlp, predict_mlp
    from mosd_hidden_dynamics.utils import save_json, source_record
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(args.run_dir).resolve()
    cfg = json.loads((root / "config_requested.json").read_text())
    extra = json.loads((repo / "configs/remaining_hidden_dynamics.json").read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    output = root / "control_mlp"
    output.mkdir(exist_ok=True)
    source = source_record(repo)
    for dataset in cfg["datasets"]:
        if dataset["family"] != "control":
            continue
        for protocol in ("in_series", "out_of_series"):
            directory = output / f"{dataset['name']}_{protocol}"
            if (directory / "complete.json").exists():
                continue
            directory.mkdir(exist_ok=True)
            started = time.perf_counter()
            group, anchors, digest = load_extended_group(cfg, registry, root, dataset["name"], protocol)
            train, val = group["train"], group["val"]
            write_rows(directory / "anchors.csv", anchors)
            rows = []
            for task, target in (("state", "y1"), ("depth", "depth")):
                for seed in extra["seeds"]:
                    fit = fit_mlp(train["x"], train[target], val["x"], val[target], cfg["mlp"], seed, args.device, val["ids"])
                    prediction = predict_mlp(fit, val["x"])
                    rows.extend({"dataset": dataset["name"], "family": "control", "protocol": protocol,
                                 "condition": "pretrained", "model": "mlp_" + task, "horizon": 1, "seed": seed, **r} for r in
                                series_metrics(val["y1"], prediction, train["y1"].mean(0), val["ids"]))
                    torch.save({"state_dict": fit["model"].state_dict(), "config": cfg["mlp"],
                                **{key: fit[key] for key in ("x_mean", "x_scale", "y_mean", "y_scale")}},
                               directory / f"mlp_{task}_seed{seed}.pt")
                    save_json(directory / f"fit_{task}_seed{seed}.json",
                              {key: fit[key] for key in ("seed", "history", "seconds", "best_epoch", "parameters")})
                    print(dataset["name"], protocol, task, seed, fit["best_epoch"], flush=True)
            save_json(directory / "metrics.json", rows)
            write_rows(directory / "metrics.csv", rows)
            save_json(directory / "run_metadata.json", {"source": source, "anchor_sha256": digest, "test_evaluated": False})
            save_json(directory / "complete.json", {"seconds": time.perf_counter() - started, "test_evaluated": False})
    save_json(output / "complete.json", {"groups": 4, "fits": 24, "test_evaluated": False})


if __name__ == "__main__":
    main()
