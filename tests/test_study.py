import copy
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosd_hidden_dynamics.study_models import fit_mlp, fit_ridge, predict_mlp, predict_ridge, prepare_ridge, rollout_ridge
from mosd_hidden_dynamics.study_data import causal_prefix, study_pairs


class StudyModelTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
