import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Direct nonlinear forecasts at the remaining long horizons")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--evaluation-id", default="main")
    parser.add_argument("--device", default="cuda:7")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mosd_hidden_dynamics.study import load_group, write_rows
    from mosd_hidden_dynamics.study_models import diagnostic_metrics, fit_mlp, predict_mlp
    from mosd_hidden_dynamics.utils import save_json, source_record
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(args.run_dir).resolve()
    cfg = json.loads((root / "config_requested.json").read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    output = root / "evaluations" / args.evaluation_id / "long_mlp"
    output.mkdir(exist_ok=False)
    save_json(output / "run_metadata.json", {"source": source_record(repo), "config": cfg["mlp"],
              "horizons": [k for k in cfg["horizons"] if k not in cfg["mlp"]["horizons"]], "test_evaluated": False})
    rows = []
    for dataset in cfg["datasets"]:
        if dataset["family"] == "control":
            continue
        group, _ = load_group(cfg, registry, root, dataset["name"], "pretrained", "out_of_series")
        train, val = group["train"], group["val"]
        for k in cfg["horizons"]:
            if k in cfg["mlp"]["horizons"]:
                continue
            for seed in cfg["mlp"]["seeds"]:
                fit = fit_mlp(train["x"], train[f"y{k}"], val["x"], val[f"y{k}"], cfg["mlp"], seed, args.device, val["ids"])
                prediction = predict_mlp(fit, val["x"])
                for series in np.unique(val["ids"]):
                    mask = val["ids"] == series
                    rows.append({"dataset": dataset["name"], "family": dataset["family"], "protocol": "out_of_series",
                                 "condition": "pretrained", "model": "mlp_direct", "horizon": k, "seed": seed,
                                 "alpha": None, "series_id": series, "n_train": len(train["x"]), "n_val": int(mask.sum()),
                                 **diagnostic_metrics(val[f"y{k}"][mask], prediction[mask], train[f"y{k}"].mean(axis=0))})
                label = f"{dataset['name']}_k{k}_seed{seed}"
                torch.save({"state_dict": fit["model"].state_dict(), "x_mean": torch.from_numpy(fit["x_mean"]),
                            "x_scale": torch.from_numpy(fit["x_scale"]), "y_mean": torch.from_numpy(fit["y_mean"]),
                            "y_scale": torch.from_numpy(fit["y_scale"]), "config": cfg["mlp"]}, output / (label + ".pt"))
                save_json(output / (label + ".json"), {key: fit[key] for key in ("best_epoch", "history", "seconds", "parameters", "seed")})
                save_json(output / "metrics.json", rows)
                current = [r for r in rows if r["dataset"] == dataset["name"] and r["horizon"] == k and r["seed"] == seed]
                print(f"{label}: R2={np.mean([r['r2'] for r in current]):.3f}, epoch={fit['best_epoch']}, seconds={fit['seconds']:.2f}", flush=True)
        del group
        torch.cuda.empty_cache()
    write_rows(output / "metrics.csv", rows)
    save_json(output / "complete.json", {"trained_models": len(rows) // 4, "test_evaluated": False})


if __name__ == "__main__":
    main()
