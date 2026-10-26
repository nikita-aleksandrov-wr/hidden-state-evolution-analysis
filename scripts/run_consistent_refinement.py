import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Refine frozen ODE integration on matched 256-step validation paths")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--dataset", choices=("Lorenz", "Rossler", "electricity"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--stop-file", required=True)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "4"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    import numpy as np
    import torch
    from mosd_hidden_dynamics.consistent import load_transition, transition, path_statistics, score_series
    from mosd_hidden_dynamics.utils import save_json, source_record
    from run_reencoding_control import cached_states
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(args.run_dir).resolve()
    directory = root / "core" / args.dataset
    assert json.loads((directory / "complete.json").read_text())["status"] == "complete"
    anchors = json.loads((directory / "report_anchors.json").read_text())
    with np.load(directory / "observed_paths.npz") as data:
        indices, truth = data["indices"], data["paths"]
    selected = [anchors[index] for index in indices]
    initial = cached_states(Path(args.reference_run), selected, "pretrained").astype(np.float64)
    with np.load(directory / "scoring_space.npz") as data:
        space = dict(data)
    with np.load(directory / "report_targets.npz") as data:
        ids = data["ids"][indices]
        targets = {k: data[f"y{k}"][indices] for k in (1, 4, 16, 64, 256)}
        references = {k: data[f"reference{k}"] for k in targets}
    np.testing.assert_allclose((initial - space["mean"]) @ space["vectors"][:, :8], truth[:, 0], rtol=1e-6, atol=1e-6)
    source = source_record(repo)
    for seed in args.seeds:
        out = root / "refinement" / args.dataset / str(seed)
        out.mkdir(parents=True, exist_ok=True)
        if (out / "complete.json").exists():
            continue
        fit = load_transition(directory / f"ode_{seed}.pt", args.device)
        all_predictions, rows, statistics = {}, [], []
        began = time.perf_counter()
        for substeps in (4, 8, 16):
            path, predictions = [truth[:, 0]], {}
            with torch.inference_mode():
                state = torch.as_tensor((initial - fit["mean"]) / fit["scale"], device=args.device, dtype=torch.float32)
                for k in range(1, 257):
                    if Path(args.stop_file).exists():
                        raise InterruptedError("Stop requested; other completed seeds remain saved")
                    state = transition(fit["model"], state, "ode", substeps)
                    raw = state.cpu().numpy().astype(np.float64) * fit["scale"] + fit["mean"]
                    path.append((raw - space["mean"]) @ space["vectors"][:, :8])
                    if k in targets:
                        predictions[k] = raw
            path = np.asarray(path).transpose(1, 0, 2)
            assert np.isfinite(path).all()
            all_predictions[substeps] = predictions
            for k, value in predictions.items():
                rows.extend({"dataset": args.dataset, "model": "ode", "seed": seed, "substeps": substeps,
                             "horizon": k, **row} for row in score_series(targets[k], value, references[k], ids, space))
            statistics.append({"dataset": args.dataset, "seed": seed, "substeps": substeps,
                               **path_statistics(path, truth, space["pc_scale"][:8])})
            np.savez_compressed(out / f"paths_substeps{substeps}.npz", paths=path)
            print(args.dataset, seed, "substeps", substeps, "complete", flush=True)
        differences = [{"horizon": k, "relative_difference_8_vs_16":
                        float(np.linalg.norm(all_predictions[8][k] - all_predictions[16][k]) /
                              max(np.linalg.norm(all_predictions[16][k]), 1e-12))} for k in targets]
        save_json(out / "metrics.json", rows)
        save_json(out / "path_statistics.json", statistics)
        save_json(out / "resolution.json", differences)
        save_json(out / "complete.json", {"status": "complete", "source": source, "test_evaluated": False,
            "seconds": time.perf_counter() - began, "selected_indices": indices.tolist(),
            "interpretation": "Solver sensitivity of one frozen field on fixed validation starts; no retraining or new model selection"})


if __name__ == "__main__":
    main()
