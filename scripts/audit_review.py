import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Check completed review experiments, shared anchors and reserved data")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--additional-run", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mosd_hidden_dynamics.utils import save_json, sha256_file
    root, extra = Path(args.run_dir).resolve(), Path(args.additional_run).resolve()
    counts = {}
    for folder, expected in (("remaining_baselines", 10), ("neural_ode", 6),
                             ("residual_training", 3), ("ode_full_rollouts", 3),
                             ("ablations/evaluations", 244), ("control_mlp", 4), ("ode_resolution", 6)):
        paths = sorted((root / folder).glob("*/complete.json"))
        assert len(paths) == expected, (folder, len(paths), expected)
        counts[folder] = len(paths)
        for path in paths:
            assert not json.loads(path.read_text())["test_evaluated"]
            for row in json.loads((path.parent / "metrics.json").read_text()):
                assert np.isfinite(row["r2"]) and row["r2"] <= 1 + 1e-8
                assert np.isfinite(row["cosine"]) and abs(row["cosine"]) <= 1 + 1e-8
    for name in ("ode_interpolation", "research_controls"):
        assert not json.loads((root / name / "complete.json").read_text())["test_evaluated"]
    assert json.loads((root / "chaotic_regimes/complete.json").read_text())["no_model_test_data_used"]
    root_cases = sorted((root / "matched_roots").glob("*/complete.json"))
    assert len(root_cases) == 6
    counts["matched_roots"] = len(root_cases)
    for path in root_cases:
        assert not json.loads(path.read_text())["test_evaluated"]
        metadata = json.loads((path.parent / "run_metadata.json").read_text())
        assert metadata["starts"] == 50 and not metadata["refitted"]
        for seed in (2026, 2027, 2028):
            diagnostics = json.loads((path.parent / f"roots_seed{seed}.json").read_text())
            assert len(diagnostics["candidates"]) == 50
    main_registry = json.loads((root / "series_registry.json").read_text())
    extra_registry = json.loads((extra / "series_registry.json").read_text())
    reserved_count, feature_count, anchor_count = 0, 0, 0
    for run, registry, feature_roots in ((root, main_registry, [root / "ablations/features"]),
                                        (extra, extra_registry, [extra / "features"])):
        records = {record["series_id"]: record for record in registry}
        context = json.loads((run / "config_requested.json").read_text())["context_length"]
        for record in registry:
            raw_path = run / "data" / (record["series_id"] + ".npz")
            if record["role"] == "test":
                reserved_count += 1
                assert not raw_path.exists()
            else:
                assert sha256_file(raw_path) == record["data_sha256"]
        for feature_root in feature_roots:
            for path in feature_root.rglob("*.npz"):
                record = records[path.stem]
                assert record["role"] != "test"
                meta = json.loads(path.with_suffix(".json").read_text())
                assert not meta["test_states_extracted"] and not meta["future_values_supplied"]
                assert sha256_file(path) == meta["sha256"]
                assert meta["data_sha256"] == record["data_sha256"]
                with np.load(path) as data:
                    assert np.isfinite(data["states"]).all()
                    assert np.all(data["raw_endpoints"] < record["test_start"])
                feature_count += 1
        anchors = list(run.rglob("*anchors.csv")) if run == extra else [
            p for name in ("remaining_baselines", "neural_ode", "residual_training", "ode_full_rollouts", "ablations/evaluations")
            for p in (run / name).rglob("*anchors.csv")]
        for path in anchors:
            metadata_path = path.parent / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            protocol = metadata.get("case", {}).get("protocol")
            if protocol is None:
                protocol = "in_series" if "_in_series" in str(path.parent) else "out_of_series"
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                record = records[row["series_id"]]
                assert record["role"] != "test" and int(row["max_target_end"]) < record["test_start"]
                assert int(row["receptive_start"]) >= 0
                if protocol == "out_of_series":
                    assert row["split"] == record["role"]
                else:
                    boundary = int(record["length"] * .7)
                    if row["split"] == "train":
                        assert int(row["max_target_end"]) < boundary
                    else:
                        assert int(row["receptive_start"]) >= boundary + context
                anchor_count += 1
    dense_precision = []
    for dataset in ("Lorenz", "Rossler", "electricity"):
        for protocol in ("in_series", "out_of_series"):
            case = f"{dataset}_{protocol}"
            baseline = json.loads((root / "remaining_baselines" / case / "run_metadata.json").read_text())
            ode = json.loads((root / "neural_ode" / case / "run_metadata.json").read_text())
            dense = json.loads((root / "ode_interpolation" / case / "run_metadata.json").read_text())
            refined = json.loads((root / "ode_resolution" / case / "run_metadata.json").read_text())
            assert baseline["anchor_sha256"] == ode["anchor_sha256"] == dense["anchor_sha256"]
            assert refined["anchor_sha256"] == baseline["anchor_sha256"] and not refined["refitted"]
            assert 0 <= dense["coarse_state_fp16_value_match_fraction"] <= 1
            for path in (root / "ode_interpolation" / case).glob("*_dense.npz"):
                series = path.stem.removesuffix("_dense")
                with np.load(path) as data, np.load(root / "features/pretrained" / (series + ".npz")) as coarse:
                    common, fresh_index, cached_index = np.intersect1d(
                        data["raw_endpoints"], coarse["raw_endpoints"], return_indices=True)
                    assert len(common) > 0
                    fresh, cached = data["states"][fresh_index], coarse["states"][cached_index, 0].astype(np.float32)
                    relative = float(np.linalg.norm(fresh.astype(np.float64) - cached) / np.linalg.norm(cached))
                    assert relative < np.finfo(np.float16).eps, (case, series, relative)
                    dense_precision.append(relative)
            for filename, digest in dense["frozen_fits"].items():
                assert sha256_file(root / filename) == digest
        residual = json.loads((root / "residual_training" / dataset / "run_metadata.json").read_text())
        long = json.loads((root / "ode_full_rollouts" / dataset / "run_metadata.json").read_text())
        assert residual["evaluation_anchor_sha256"] == long["anchor_sha256"]
    assert len(list((extra / "evaluations/main").glob("*/complete.json"))) == 8
    assert not json.loads((extra / "frozen_transfer/complete.json").read_text())["test_evaluated"]
    result = {"status": "passed", "groups": counts, "additional_groups": 8,
              "new_feature_files_verified": feature_count, "anchor_rows_checked": anchor_count,
              "dense_vs_cached_max_relative_l2": max(dense_precision),
              "reserved_entities_not_loaded": reserved_count, "test_evaluated": False}
    save_json(root / "audit_review.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
