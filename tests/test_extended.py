import unittest

import numpy as np

from mosd_hidden_dynamics.extended import (dmd_rollout, fit_dmd, fit_pca, fit_var,
    matched_mlp_config, project, reconstruct, var_rollout)


class ExtendedModelsTests(unittest.TestCase):
    def test_position_control_is_past_only_and_restores_encoder(self):
        import torch
        from chronos.chronos2.config import Chronos2CoreConfig
        from chronos.chronos2.model import Chronos2Model
        from mosd_hidden_dynamics.ablations import extract_variant
        from mosd_hidden_dynamics.extraction import extract_vectors
        torch.manual_seed(19)
        config = Chronos2CoreConfig(d_model=8, d_kv=4, d_ff=16, num_layers=2, num_heads=2,
            dropout_rate=0, attn_implementation="eager", chronos_config={"context_length": 32,
            "input_patch_size": 4, "input_patch_stride": 4, "output_patch_size": 4,
            "quantiles": [.1, .5, .9], "use_reg_token": True})
        model = Chronos2Model(config).eval()
        observed = np.random.default_rng(20).normal(size=64).astype(np.float32)
        before = extract_vectors(model, observed, np.array([31, 39]), [0, 1], 32, 2)
        positions = []
        handle = model.encoder.block[0].layer[0].self_attention.register_forward_pre_hook(
            lambda module, args, kwargs: positions.append(kwargs["position_ids"].detach().cpu().numpy().copy()),
            with_kwargs=True)
        shuffled = extract_variant(model, observed, np.array([31, 39]), 32, 2, [0, 1], 2026)
        altered = observed.copy()
        altered[40:] = 10000
        repeated = extract_variant(model, altered, np.array([31, 39]), 32, 2, [0, 1], 2026)
        handle.remove()
        after = extract_vectors(model, observed, np.array([31, 39]), [0, 1], 32, 2)
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(shuffled, repeated)
        np.testing.assert_array_equal(np.sort(positions[0][0, :8]), np.arange(8))
        np.testing.assert_array_equal(positions[0][0, 8:], [8, 9])
        self.assertFalse(np.array_equal(positions[0][0, :8], np.arange(8)))
        self.assertEqual(shuffled.shape, (2, 3, 8))

    def test_context_ablations_share_endpoints_and_preserve_entity_holdout(self):
        from pathlib import Path
        import tempfile
        import json
        from mosd_hidden_dynamics.ablations import load_ablation_group
        from mosd_hidden_dynamics.utils import sha256_file
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = []
            for series, role in [("a", "train"), ("b", "val")]:
                record = {"series_id": series, "role": role, "dataset": "toy", "length": 4096,
                          "test_start": 3276, "data_sha256": "fixed_raw_digest"}
                registry.append(record)
                for context in (16, 32):
                    output = root / "ablations/features" / f"context_{context}"
                    output.mkdir(parents=True, exist_ok=True)
                    ends = np.arange(context - 1, 3276, 16)
                    states = np.tile(ends[:, None, None], (1, 1, 3)).astype(np.float32)
                    path = output / (series + ".npz")
                    np.savez(path, states=states, raw_endpoints=ends, observed=np.arange(3276))
                    path.with_suffix(".json").write_text(json.dumps({"spec": {"context": context},
                        "sha256": sha256_file(path), "data_sha256": record["data_sha256"],
                        "labels": ["block5"], "test_states_extracted": False}))
            manifests = []
            for context in (16, 32):
                group, anchors, _ = load_ablation_group({}, registry, root, "toy", "out_of_series",
                    f"context_{context}", "block5", common_context=64)
                self.assertEqual(set(group["train"]["ids"]), {"a"})
                self.assertEqual(set(group["val"]["ids"]), {"b"})
                np.testing.assert_array_equal(group["val"]["y4"] - group["val"]["x"], 64)
                manifests.append(anchors)
            self.assertEqual(manifests[0], manifests[1])

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
