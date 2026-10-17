import json
import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosd_hidden_dynamics.study_models import fit_mlp, fit_ridge, predict_mlp, predict_ridge, prepare_ridge, rollout_ridge
from mosd_hidden_dynamics.study_models import load_mlp, nearest_indices
from mosd_hidden_dynamics.study_data import causal_prefix, study_pairs
from mosd_hidden_dynamics.study import aggregate_results, depth_comparison, load_group
from mosd_hidden_dynamics.utils import sha256_file


class StudyModelTests(unittest.TestCase):
    def test_raw_neighbors_use_training_windows_and_ignore_offset_and_scale(self):
        train = np.random.default_rng(51).normal(size=(12, 16))
        val = train[[3, 7]] * 2 + 10
        indices, similarity = nearest_indices(train, val, 5, "cpu")
        np.testing.assert_array_equal(indices[:, 0], [3, 7])
        np.testing.assert_allclose(similarity[:, 0], 1, atol=1e-6)
        self.assertTrue(np.all(indices < len(train)))

    def test_new_series_holdout_and_all_horizons_use_the_same_anchors(self):
        rng = np.random.default_rng(49)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "features/pretrained").mkdir(parents=True)
            (root / "data").mkdir()
            registry = []
            for series, role in (("a", "train"), ("b", "val")):
                path = root / "data" / (series + ".npz")
                np.savez(path, observed=rng.normal(size=4096).astype(np.float32))
                record = {"series_id": series, "dataset": "synthetic", "role": role, "length": 4096,
                          "test_start": 3276, "data_sha256": sha256_file(path)}
                registry.append(record)
                ends = np.arange(15, 3276, 16)
                states = np.stack([np.tile(ends[:, None], (1, 3)), np.tile(ends[:, None], (1, 3)) + 1], axis=1)
                feature = root / "features/pretrained" / (series + ".npz")
                np.savez(feature, states=states, raw_endpoints=ends)
                feature.with_suffix(".json").write_text(json.dumps({"sha256": sha256_file(feature),
                    "data_sha256": record["data_sha256"], "test_states_extracted": False, "future_values_supplied": False}))
            registry.append({"series_id": "reserved", "dataset": "synthetic", "role": "test"})
            group, manifest = load_group({"context_length": 16, "horizons": [1, 4]}, registry, root, "synthetic", "pretrained", "out_of_series")
            self.assertEqual(set(group["train"]["ids"]), {"a"})
            self.assertEqual(set(group["val"]["ids"]), {"b"})
            for split in group.values():
                np.testing.assert_array_equal(split["y4"] - split["x"], 64)
                np.testing.assert_array_equal(split["y1"] - split["x"], 16)
            self.assertNotIn("reserved", {row["series_id"] for row in manifest})

    def test_bootstrap_pairs_series_and_averages_seeds_before_series(self):
        rows = []
        for series in ("a", "b", "c"):
            common = {"dataset": "example", "protocol": "out_of_series", "condition": "pretrained",
                      "horizon": 1, "series_id": series, "centered_cosine": .3}
            rows.extend([{**common, "model": "persistence", "seed": None, "r2": .1, "cosine": .2},
                         {**common, "model": "ridge_depth", "seed": None, "r2": .2, "cosine": .3},
                         {**common, "model": "ridge_direct", "seed": None, "r2": .8, "cosine": .9}])
            for seed, score in ((1, .7), (2, .9)):
                rows.append({**common, "model": "mlp_direct", "seed": seed, "r2": score, "cosine": .9})
        summaries = aggregate_results(rows, 100, 50)
        mlp = next(r for r in summaries if r["model"] == "mlp_direct")
        self.assertAlmostEqual(mlp["r2"], .8)
        self.assertAlmostEqual(mlp["delta_r2_ci_low"], .7)
        self.assertEqual(mlp["n_seeds"], 2)
        self.assertTrue(depth_comparison(rows, 100, 50)[0]["local_threshold_met"])

    def test_missing_values_are_filled_only_from_the_past(self):
        values = np.array([np.nan, 2., np.nan, 5., np.nan, 9.])
        observed, offset, count = causal_prefix(values, 5)
        np.testing.assert_array_equal(observed, [2., 2., 5., 5., 9.])
        self.assertEqual((offset, count), (1, 2))
        changed = values.copy()
        changed[-1] = -100
        np.testing.assert_array_equal(causal_prefix(changed, 5)[0][:-1], observed[:-1])

    def test_common_anchors_respect_context_memory_horizon_and_test(self):
        record = {"role": "val", "length": 32768, "test_start": int(32768 * .8)}
        ends = np.arange(511, record["test_start"], 16)
        pairs = study_pairs(ends, record, 512, 64, "in_series")
        for split, indices in pairs.items():
            start = 0 if split == "train" else int(32768 * .7) + 512
            stop = int(32768 * .7) if split == "train" else record["test_start"]
            self.assertTrue(np.all(ends[indices] - 16 - 511 >= start))
            self.assertTrue(np.all(ends[indices] + 64 * 16 < stop))
        self.assertEqual(set(study_pairs(ends, record, 512, 64, "out_of_series")), {"val"})
        with self.assertRaisesRegex(ValueError, "reserved"):
            study_pairs(ends, {**record, "role": "test"}, 512, 64, "out_of_series")

    def test_ridge_matches_sklearn_with_train_only_scaling(self):
        rng = np.random.default_rng(41)
        x, y, val = rng.normal(size=(40, 5)), rng.normal(size=(40, 3)), rng.normal(size=(8, 5)) + 7
        x[:, 4], val[:, 4] = 3, 3
        prepared = prepare_ridge(x, "cpu")
        scaler = StandardScaler().fit(x)
        for alpha in (0.0001, 1, 100):
            actual = predict_ridge(fit_ridge(prepared, y, alpha), val)
            reference = Ridge(alpha=alpha, solver="svd").fit(scaler.transform(x), y).predict(scaler.transform(val))
            np.testing.assert_allclose(actual, reference, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(prepared["mean"].numpy(), x.mean(axis=0))

    def test_recursive_forecast_uses_its_own_previous_output(self):
        import torch
        a = np.array([[1.1, 0.2], [-0.1, 0.8]])
        b, x = np.array([0.1, 0.3]), np.array([[2., 1.], [3., -1.]])
        model = {"coef": torch.tensor(a), "intercept": torch.tensor(b),
                 "mean": torch.zeros(2, dtype=torch.float64), "scale": torch.ones(2, dtype=torch.float64)}
        result = rollout_ridge(model, x, [1, 4, 16])
        expected = x.copy()
        for step in range(1, 17):
            expected = expected @ a.T + b
            if step in result:
                np.testing.assert_allclose(result[step], expected, atol=1e-12)
        np.testing.assert_array_equal(x, [[2., 1.], [3., -1.]])

    def test_mlp_scaling_and_deterministic_seed(self):
        import torch
        torch.set_num_threads(1)
        rng = np.random.default_rng(43)
        x, val = rng.normal(size=(128, 3)), rng.normal(size=(24, 3))
        weight = rng.normal(size=(3, 2))
        y, target = x @ weight + 2, val @ weight + 2
        cfg = {"width": 16, "learning_rate": 0.01, "weight_decay": 0.,
               "epochs": 35, "patience": 8, "batch_size": 64}
        first = fit_mlp(x, y, val, target, cfg, 44, "cpu")
        second = fit_mlp(x, y, val, target, cfg, 44, "cpu")
        np.testing.assert_array_equal(predict_mlp(first, val), predict_mlp(second, val))
        np.testing.assert_allclose(first["x_mean"], x.mean(axis=0))
        np.testing.assert_allclose(first["y_mean"], y.mean(axis=0))
        self.assertLess(np.square(predict_mlp(first, val) - target).mean(), np.square(y.mean(axis=0) - target).mean() * 0.1)
        with tempfile.TemporaryDirectory() as temp:
            for tensors in (False, True):
                saved = {"state_dict": first["model"].state_dict(), "config": cfg}
                for name in ("x_mean", "x_scale", "y_mean", "y_scale"):
                    saved[name] = torch.from_numpy(first[name]) if tensors else first[name]
                path = Path(temp) / "model.pt"
                torch.save(saved, path)
                np.testing.assert_array_equal(predict_mlp(load_mlp(path, "cpu"), val), predict_mlp(first, val))


if __name__ == "__main__":
    unittest.main()
