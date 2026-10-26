import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="Known-flow positive control for the matched Neural ODE training pipeline")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--stop-file", required=True)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    import numpy as np
    import torch
    from mosd_hidden_dynamics.consistent import fit_transition, save_transition, transition
    from mosd_hidden_dynamics.extended import series_metrics
    from mosd_hidden_dynamics.utils import save_json, source_record
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.output_dir).resolve() / "positive_control"
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((repo / "configs/consistent_hidden_dynamics.json").read_text())["training"]
    matrix = torch.tensor([[-.02, -.2], [.2, -.02]], dtype=torch.float64)
    step = torch.matrix_exp(matrix).numpy()
    rng = np.random.default_rng(2026)
    def group(n):
        initial = rng.uniform(-2, 2, size=(n, 2)).astype(np.float32)
        paths = np.stack([initial @ np.linalg.matrix_power(step, k).T for k in range(5)], axis=1).astype(np.float32)
        return {"x": initial, "trajectory": paths, "y4": paths[:, 4], "ids": np.repeat("oscillator", n)}
    train, val = group(1024), group(256)
    fit = fit_transition(train, val, cfg, "ode", 2026, args.device, Path(args.stop_file))
    save_transition(out / "ode.pt", fit)
    predictions = {}
    with torch.inference_mode():
        for substeps in (4, 8, 16):
            z = torch.as_tensor((val["x"] - fit["mean"]) / fit["scale"], device=args.device)
            for _ in range(4):
                z = transition(fit["model"], z, "ode", substeps)
            predictions[substeps] = z.cpu().numpy() * fit["scale"] + fit["mean"]
    scores = series_metrics(val["y4"], predictions[16], train["y4"].mean(0), val["ids"])[0]
    difference = float(np.linalg.norm(predictions[8] - predictions[16]) / np.linalg.norm(predictions[16]))
    result = {"status": "passed" if scores["r2"] >= .95 and difference < .001 else "failed_accuracy_or_resolution",
              "source": source_record(repo), "matrix": matrix.tolist(), "validation_metrics": scores,
              "relative_difference_8_vs_16": difference, "training_seconds": fit["seconds"],
              "criterion": "R2 >= 0.95 and relative 8-versus-16-substep prediction difference < 0.001 at k4",
              "test_evaluated": False, "interpretation": "Implementation control on a known stable oscillator, not a hidden-state result"}
    save_json(out / "result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "validation_metrics", "relative_difference_8_vs_16", "training_seconds")}), flush=True)


if __name__ == "__main__":
    main()
