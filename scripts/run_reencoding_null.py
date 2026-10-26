import argparse
import copy
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="Check whether a temporal ridge beats input re-encoding even on white noise")
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    import numpy as np
    import torch
    from mosd_hidden_dynamics.consistent import score_series, scoring_space
    from mosd_hidden_dynamics.extended import load_extended_group, series_metrics
    from mosd_hidden_dynamics.study_models import fit_ridge, prepare_ridge, rollout_ridge, save_ridge
    from mosd_hidden_dynamics.utils import save_json, source_record
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(args.reference_run).resolve()
    out = Path(args.output_dir).resolve() / "reencoding/white_noise"
    cfg = json.loads((repo / "configs/consistent_hidden_dynamics.json").read_text())
    base = json.loads((root / "config_requested.json").read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    short, _, _ = load_extended_group(base, registry, root, "white_noise", "out_of_series", order=8, segment=4)
    long_cfg = copy.deepcopy(base)
    long_cfg["horizons"] = cfg["horizons"]
    long, _, digest = load_extended_group(long_cfg, registry, root, "white_noise", "out_of_series", order=8, segment=4)
    with np.load(out / "pretrained_predictions.npz") as arrays:
        indices = arrays["selected_indices"].copy()
    train, val = short["train"], short["val"]
    prepared = prepare_ridge(train["x"], args.device)
    best, choices = None, []
    for alpha in cfg["ridge_alphas"]:
        fit = fit_ridge(prepared, train["y1"], alpha)
        prediction = rollout_ridge(fit, val["x"], [4])[4]
        score = float(np.mean([r["r2"] for r in series_metrics(val["y4"], prediction, train["y4"].mean(0), val["ids"])]))
        choices.append({"alpha": alpha, "validation_macro_r2_k4": score})
        if best is None or (score, -alpha) > best[:2]:
            best = score, -alpha, fit
    fit = best[2]
    save_ridge(out / "null_ridge.npz", fit)
    predictions = rollout_ridge(fit, long["val"]["x"][indices], cfg["plugin_horizons"])
    space = scoring_space(train["x"], args.device)
    rows = []
    for horizon, prediction in predictions.items():
        rows.extend({"dataset": "white_noise", "method": "ridge", "horizon": horizon, **r}
                    for r in score_series(long["val"][f"y{horizon}"][indices], prediction,
                        long["train"][f"y{horizon}"].mean(0), long["val"]["ids"][indices], space))
    save_json(out / "null_ridge_metrics.json", rows)
    save_json(out / "null_ridge_metadata.json", {"source": source_record(repo), "report_anchor_sha256": digest,
        "selection": choices, "test_evaluated": False,
        "interpretation": "Beating the plug-in baseline is not by itself evidence of meaningful autonomous dynamics."})
    print("White-noise readout specificity check complete", flush=True)


if __name__ == "__main__":
    main()
