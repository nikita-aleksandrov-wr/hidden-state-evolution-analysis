import argparse
import csv
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Audit study artifacts, held-out memberships and numerical agreement")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--evaluation-id", default="main")
    parser.add_argument("--verify-ridge", action="store_true")
    parser.add_argument("--device", default="cuda:7")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "4"
    import numpy as np
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mosd_hidden_dynamics.study import load_group
    from mosd_hidden_dynamics.study_models import load_ridge, predict_ridge
    from mosd_hidden_dynamics.utils import save_json, sha256_file
    root = Path(args.run_dir).resolve()
    cfg = json.loads((root / "config_requested.json").read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    evaluation = root / "evaluations" / args.evaluation_id
    conditions = ["pretrained"] + [f"random_{s}" for s in cfg["random_init_seeds"]]
    active = [r for r in registry if r["role"] != "test"]
    reserved = [r for r in registry if r["role"] == "test"]
    for record in registry:
        path = root / "data" / (record["series_id"] + ".npz")
        if record["role"] == "test":
            assert not path.exists(), "Reserved series was generated"
        else:
            assert sha256_file(path) == record["data_sha256"], "Raw data changed"
    feature_count = 0
    for condition in conditions:
        for record in registry:
            path = root / "features" / condition / (record["series_id"] + ".npz")
            if record["role"] == "test":
                assert not path.exists(), "Reserved series was encoded"
                continue
            meta = json.loads(path.with_suffix(".json").read_text())
            assert not meta["test_states_extracted"] and not meta["future_values_supplied"]
            assert meta["sha256"] == sha256_file(path) and meta["data_sha256"] == record["data_sha256"]
            with np.load(path) as arrays:
                assert arrays["states"].shape[1:] == (2, 768)
                assert arrays["states"].dtype == np.float16 and np.isfinite(arrays["states"]).all()
                assert np.all(arrays["raw_endpoints"] < record["test_start"])
                np.testing.assert_array_equal(arrays["layers"], [5, 6])
            feature_count += 1
    snapshots = []
    for condition in conditions:
        with np.load(root / "features" / condition / (active[0]["series_id"] + ".npz")) as data:
            snapshots.append(data["states"][:8].copy())
    for index, first in enumerate(snapshots):
        assert all(not np.array_equal(first, other) for other in snapshots[index + 1:]), "Initialization conditions coincide"
    groups, metric_rows, null_fits, mlp_fits = 0, 0, 0, 0
    for condition in conditions:
        for dataset in cfg["datasets"]:
            for protocol in ("in_series", "out_of_series"):
                directory = evaluation / f"{condition}_{dataset['name']}_{protocol}"
                complete = json.loads((directory / "complete.json").read_text())
                assert not complete["test_evaluated"]
                with (directory / "anchors.csv").open() as f:
                    anchors = list(csv.DictReader(f))
                declared = {r["series_id"]: r for r in registry if r["dataset"] == dataset["name"] and r["role"] != "test"}
                for row in anchors:
                    record = declared[row["series_id"]]
                    assert int(row["max_target_end"]) < record["test_start"]
                    assert int(row["receptive_start"]) >= 0
                    if protocol == "out_of_series":
                        assert row["split"] == record["role"]
                    else:
                        boundary = int(record["length"] * .7)
                        if row["split"] == "train":
                            assert int(row["max_target_end"]) < boundary
                        else:
                            assert int(row["receptive_start"]) >= boundary + cfg["context_length"]
                metrics = json.loads((directory / "metrics.json").read_text())
                for row in metrics:
                    assert row["series_id"] in declared
                    assert np.isfinite(row["r2"]) and row["r2"] <= 1 + 1e-8
                    assert -1 - 1e-8 <= row["cosine"] <= 1 + 1e-8
                    if row["model"] == "train_mean":
                        assert abs(row["r2"]) < 1e-10
                if condition == "pretrained":
                    controls = json.loads((directory / "null_controls.json").read_text())
                    for kind in ("shuffle", "circular_shift"):
                        seeds = {r["seed"] for r in controls if r["control"] == kind}
                        assert seeds == set(range(cfg["seed"], cfg["seed"] + cfg["shuffled_seeds"]))
                        null_fits += len(seeds)
                    mlp_fits += len(list(directory.glob("mlp_*.pt")))
                metric_rows += len(metrics)
                groups += 1
    for name in ("long_mlp", "transfer", "analogs", "long_rollouts"):
        assert not json.loads((evaluation / name / "complete.json").read_text())["test_evaluated"]
    mlp_fits += len(list((evaluation / "long_mlp").glob("*.pt")))
    report = {"status": "passed", "generated_series": len(active), "reserved_series_not_loaded": len(reserved),
              "feature_files": feature_count, "condition_protocol_dataset_groups": groups,
              "metric_rows": metric_rows, "null_control_fits": null_fits, "mlp_fits": mlp_fits,
              "test_evaluated": False}
    if args.verify_ridge:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        group, _ = load_group(cfg, registry, root, "Lorenz", "pretrained", "out_of_series")
        train, val = group["train"], group["val"]
        directory = evaluation / "pretrained_Lorenz_out_of_series"
        alpha = json.loads((directory / "linear_diagnostics.json").read_text())["alpha"]
        scaler = StandardScaler().fit(train["x"])
        reference = Ridge(alpha=alpha, solver="svd").fit(scaler.transform(train["x"]), train["y1"])
        expected = reference.predict(scaler.transform(val["x"]))
        actual = predict_ridge(load_ridge(directory / "ridge_k1.npz", args.device), val["x"])
        relative = float(np.linalg.norm(expected - actual) / np.linalg.norm(expected))
        assert relative < 1e-6, "GPU ridge differs from the reference SVD solver"
        report["ridge_relative_difference_vs_sklearn_svd"] = relative
    save_json(root / "audit_study.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
