import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Finite-time Lyapunov checks of the actual synthetic-system regimes")
    parser.add_argument("--main-run", required=True)
    parser.add_argument("--additional-run", required=True)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "2"
    import numpy as np
    from dysts import flows
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mosd_hidden_dynamics.chaos_checks import finite_jacobian, lyapunov_spectrum
    from mosd_hidden_dynamics.utils import save_json, source_record
    main_root, additional = Path(args.main_run).resolve(), Path(args.additional_run).resolve()
    output = main_root / "chaotic_regimes"
    output.mkdir(exist_ok=True)
    rows = []
    for root in [main_root, additional]:
        registry = json.loads((root / "series_registry.json").read_text())
        names = list(dict.fromkeys(r["dataset"] for r in registry if r["family"] == "chaotic"))
        for name in names:
            path = output / (name + ".json")
            if path.exists():
                rows.append(json.loads(path.read_text()))
                continue
            record = next(r for r in registry if r["dataset"] == name and r["role"] == "train")
            model = getattr(flows, name)()
            initial = np.asarray(record["generation"]["initial_condition"])
            if any(abs(float(model.params[k]) - value) > 1e-12 for k, value in record["generation"]["parameters"].items()):
                raise ValueError("Lyapunov check parameters differ from generated data")
            try:
                probe = np.asarray(model.jac(initial, 0))
                if probe.shape != (3, 3):
                    raise ValueError("Unexpected Jacobian")
                if not np.allclose(probe, finite_jacobian(model.rhs, initial, 0), rtol=1e-4, atol=1e-5):
                    raise ValueError("Analytic Jacobian disagrees with finite differences")
                jacobian = model.jac
                kind = "dysts analytic Jacobian"
            except (NotImplementedError, AttributeError, ValueError):
                jacobian = lambda x, t: finite_jacobian(model.rhs, x, t)
                kind = "central differences with relative coordinate step 1e-6"
            result = lyapunov_spectrum(model.rhs, jacobian, initial, model.period)
            result.update(dataset=name, jacobian=kind, parameters=record["generation"]["parameters"],
                          initial_condition=initial.tolist(), nominal_period=float(model.period),
                          largest_exponent_per_analysis_step=float(max(result["exponents_per_time"]) * record["generation"]["sampling_dt"] * 16),
                          source=source_record(repo), no_model_test_data_used=True)
            save_json(path, result)
            rows.append(result)
            print(name, result["exponents_per_time"], flush=True)
    save_json(output / "summary.json", rows)
    save_json(output / "complete.json", {"systems": len(rows), "no_model_test_data_used": True})


if __name__ == "__main__":
    main()
