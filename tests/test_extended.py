import unittest

import numpy as np

from mosd_hidden_dynamics.extended import (dmd_rollout, fit_dmd, fit_pca, fit_var,
    matched_mlp_config, project, reconstruct, var_rollout)


class ExtendedModelsTests(unittest.TestCase):
    def test_dmd_reconstructs_affine_map_and_rolls_its_own_state(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(300, 3))
        a = np.array([[.8, .1, 0], [0, .7, .1], [.05, 0, .9]])
        b = np.array([.1, -.2, .3])
        y = x @ a.T + b
        pca = fit_pca(x, 1, "cpu")
        fit = fit_dmd(pca, x, y)
        result = dmd_rollout(pca, fit, x[:4], [1, 5])
        expected = x[:4].copy()
        for _ in range(5):
            expected = expected @ a.T + b
        np.testing.assert_allclose(result[1], y[:4], atol=1e-10)
        np.testing.assert_allclose(result[5], expected, atol=1e-10)
        np.testing.assert_allclose(reconstruct(pca, project(pca, x)), x, atol=1e-10)

    def test_var_recursion_updates_both_lags(self):
        rng = np.random.default_rng(8)
        lags = rng.normal(size=(400, 2, 2))
        target = .6 * lags[:, -1] - .2 * lags[:, -2] + .1
        train = {"x": lags[:, -1], "lags": lags, "y1": target}
        val = {k: v[:20] for k, v in train.items()}
        val["ids"] = np.repeat(["a", "b"], 10)
        pca = fit_pca(train["x"], 1, "cpu")
        fit, _, _ = fit_var(pca, train, val, 2, [1e-8], "cpu")
        result = var_rollout(pca, fit, val["lags"], 2, [3])[3]
        a, b = val["lags"][:, 0].copy(), val["lags"][:, 1].copy()
        for _ in range(3):
            a, b = b, .6 * b - .2 * a + .1
        np.testing.assert_allclose(result, b, atol=1e-8)

    def test_history_mlp_matches_parameter_budget(self):
        base = {"width": 256}
        original = 256 * (768 + 768 + 1) + 768
        for width in [768, 1536, 784]:
            chosen = matched_mlp_config(base, width)["width"]
            total = chosen * (width + 768 + 1) + 768
            self.assertLess(abs(total - original) / original, .004)

    def test_rk4_matches_linear_flow_and_supports_gradients(self):
        import torch
        from mosd_hidden_dynamics.neural_ode import advance
        coefficient = torch.nn.Parameter(torch.tensor(-.3, dtype=torch.float64))
        x = torch.tensor([[1., 2.]], dtype=torch.float64)
        actual = advance(lambda z: coefficient * z, x, duration=2., substeps=16)
        expected = x * np.exp(-.6)
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)
        actual.sum().backward()
        self.assertAlmostEqual(coefficient.grad.item(), float((2 * expected).sum()), places=6)

    def test_ode_training_checkpoint_roundtrip(self):
        from pathlib import Path
        import tempfile
        from mosd_hidden_dynamics.neural_ode import fit_ode, load_ode, rollout_ode, save_ode
        rng = np.random.default_rng(12)
        x = rng.normal(size=(24, 3))
        trajectory = np.stack([x * .9 ** k for k in range(5)], axis=1)
        train = {"x": x, "trajectory": trajectory, "y4": trajectory[:, -1]}
        val = {key: value[:8] for key, value in train.items()}
        val["ids"] = np.repeat(["a", "b"], 4)
        cfg = {"width": 8, "epochs": 2, "patience": 2, "batch_size": 8, "learning_rate": .001,
               "weight_decay": .0001, "segment_steps": 4, "rk4_substeps": 1}
        fit = fit_ode(train, val, cfg, 12, "cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "own_model.pt"
            save_ode(path, fit)
            restored = load_ode(path, "cpu")
            np.testing.assert_allclose(rollout_ode(fit, x[:3], [4])[4],
                                       rollout_ode(restored, x[:3], [4])[4])


if __name__ == "__main__":
    unittest.main()
