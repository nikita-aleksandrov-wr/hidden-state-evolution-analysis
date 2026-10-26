import unittest

import numpy as np
import torch

from mosd_hidden_dynamics.consistent import fixed_starts, path_statistics, score_series, scoring_space, transition


class ConsistentDynamicsTests(unittest.TestCase):
    def test_map_residual_and_continuous_steps_have_distinct_semantics(self):
        field = torch.nn.Linear(2, 2, bias=False).double()
        matrix = torch.tensor([[-.1, -.3], [.3, -.1]], dtype=torch.float64)
        with torch.no_grad():
            field.weight.copy_(matrix)
        state = torch.tensor([[1., -.5]], dtype=torch.float64)
        torch.testing.assert_close(transition(field, state, "mlp"), state @ matrix.T)
        torch.testing.assert_close(transition(field, state, "residual"), state + state @ matrix.T)
        exact = state @ torch.matrix_exp(matrix).T
        coarse = transition(field, state, "ode", 1)
        fine = transition(field, state, "ode", 8)
        self.assertLess(float((fine - exact).norm().detach()), float((coarse - exact).norm().detach()) / 100)
        with self.assertRaises(ValueError):
            transition(field, state, "unknown")

    def test_reference_scores_and_coordinate_scaling(self):
        rng = np.random.default_rng(4)
        train = rng.normal(size=(100, 4)) * [1, 2, 4, 8]
        target = rng.normal(size=(20, 4)) * [1, 2, 4, 8]
        ids = np.repeat(["first", "second"], 10)
        reference = train.mean(0)
        space = scoring_space(train, "cpu")
        mean = score_series(target, np.broadcast_to(reference, target.shape), reference, ids, space)
        perfect = score_series(target, target, reference, ids, space)
        for row in mean:
            for metric in ("r2", "r2_standardized", "r2_whitened_pca95"):
                self.assertAlmostEqual(row[metric], 0)
        for row in perfect:
            for metric in ("r2", "r2_standardized", "r2_whitened_pca95"):
                self.assertAlmostEqual(row[metric], 1)

    def test_path_statistics_and_sampling_are_reproducible(self):
        ids = np.repeat(["a", "b"], [30, 12])
        indices = fixed_starts(ids, 16)
        np.testing.assert_array_equal(indices, fixed_starts(ids, 16))
        self.assertEqual(len(indices), 28)
        self.assertEqual(len(set(indices)), 28)
        t = np.arange(257) * .2
        paths = np.stack([np.stack([np.sin(t + p), np.cos(t + p)], axis=1) for p in [0, .3]])
        result = path_statistics(paths, paths, np.ones(2))
        self.assertAlmostEqual(result["sliced_w2_pc8_tail128"], 0)
        self.assertAlmostEqual(result["acf_mae_pc2"], 0)
        self.assertAlmostEqual(result["psd_relative_l2_pc2"], 0)

    def test_forecast_context_replacement_preserves_only_available_past(self):
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "scripts/run_reencoding_control.py"
        spec = importlib.util.spec_from_file_location("reencoding_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        past = np.arange(8)
        np.testing.assert_array_equal(module.continued_context(past, np.array([20, 21])), [2, 3, 4, 5, 6, 7, 20, 21])
        np.testing.assert_array_equal(module.continued_context(past, np.arange(100, 110)), np.arange(102, 110))
        np.testing.assert_array_equal(past, np.arange(8))


if __name__ == "__main__":
    unittest.main()
