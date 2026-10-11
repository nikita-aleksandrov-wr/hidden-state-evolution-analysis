import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosd_hidden_dynamics.baselines import fit_baselines, get_pairs, prediction_metrics
from mosd_hidden_dynamics.data import generate_lorenz
from mosd_hidden_dynamics.extraction import extract_vectors, resolve_layers
from mosd_hidden_dynamics.splits import audit_manifest, load_manifest, make_manifest, save_manifest


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke_chronos2_lorenz.json"


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(CONFIG.read_text())
        self.rows, self.bounds = make_manifest(8192, 128, 16, 1)

    def test_split_boundaries_and_receptive_fields(self):
        audit = audit_manifest(self.rows, self.bounds, np.arange(8192, dtype=np.float32))
        self.assertTrue(audit["passed"])
        self.assertEqual(self.bounds["train"][1], int(8192 * 0.7))
        self.assertEqual(self.bounds["val"][1], int(8192 * 0.8))
        self.assertTrue(all(g >= 128 for g in audit["receptive_field_gaps_raw"].values()))

    def test_wrong_target_alignment_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["target_context_end"] += 1
        with self.assertRaisesRegex(ValueError, "alignment"):
            audit_manifest(rows, self.bounds, np.arange(8192, dtype=np.float32))

    def test_exact_duplicates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit_manifest(self.rows, self.bounds, np.ones(8192, dtype=np.float32))

    def test_manifest_round_trip_and_determinism(self):
        again, _ = make_manifest(8192, 128, 16, 1)
        self.assertEqual(self.rows, again)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            save_manifest(path, self.rows)
            self.assertEqual(load_manifest(path), self.rows)

    def test_reserved_test_cannot_be_evaluated(self):
        with self.assertRaisesRegex(ValueError, "test"):
            get_pairs(self.rows, np.zeros((1, 8)), [0], "test")

    def test_r2_uses_the_training_mean(self):
        target = np.array([[2., 3.], [4., 5.], [6., 7.]])
        train_mean = np.array([1., 2.])
        perfect = prediction_metrics(target, target, train_mean)
        constant = prediction_metrics(target, np.broadcast_to(train_mean, target.shape), train_mean)
        self.assertAlmostEqual(perfect["r2"], 1)
        self.assertAlmostEqual(perfect["cosine"], 1)
        self.assertAlmostEqual(constant["r2"], 0)

    def test_input_scaler_reads_train_only(self):
        ends = sorted({r[key] for r in self.rows if r["eligible"] and r["split"] in ("train", "val")
                       for key in ("raw_context_end", "target_context_end")})
        states = np.stack([np.array([1 + e / 8192, 2 + np.sin(e / 20)]) for e in ends])
        train_x, _ = get_pairs(self.rows, states, ends, "train")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.savez(path / "states.npz", states=states[:, None], raw_endpoints=ends)
            cfg = copy.deepcopy(self.cfg)
            cfg["ridge"]["alpha_grid"] = [0.1, 1]
            metrics, _ = fit_baselines(cfg, self.rows, path)
            with np.load(path / "ridge_fit.npz") as fit:
                np.testing.assert_allclose(fit["input_mean"], train_x.mean(axis=0))
            self.assertFalse(metrics["test_evaluated"])

    def test_deterministic_official_lorenz_generation(self):
        cfg = copy.deepcopy(self.cfg["data"])
        cfg.update(trajectory_length=64, burn_in=32)
        first = generate_lorenz(cfg, 2026)
        second = generate_lorenz(cfg, 2026)
        for a, b in zip(first[:3], second[:3]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(first[0].shape, (64,))
        self.assertFalse(first[3]["standardize"])


class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from chronos.chronos2.config import Chronos2CoreConfig
        from chronos.chronos2.model import Chronos2Model

        torch.manual_seed(13)
        torch.set_num_threads(1)
        config = Chronos2CoreConfig(
            d_model=8, d_kv=4, d_ff=16, num_layers=2, num_heads=2,
            dropout_rate=0, attn_implementation="eager",
            chronos_config={"context_length": 32, "input_patch_size": 4,
                           "input_patch_stride": 4, "output_patch_size": 4,
                           "quantiles": [0.1, 0.5, 0.9], "use_reg_token": True},
        )
        cls.model = Chronos2Model(config).eval()
        cls.observed = np.random.default_rng(14).normal(size=100).astype(np.float32)

    def test_selected_vector_is_observed_patch_not_reg_or_future(self):
        captured = []
        handle = self.model.encoder.block[0].register_forward_hook(
            lambda module, args, output: captured.append(output.hidden_states.detach().numpy().copy())
        )
        try:
            result = extract_vectors(self.model, self.observed, [31], [0], 16, 1)
        finally:
            handle.remove()
        self.assertEqual(captured[0].shape, (1, 6, 8))
        np.testing.assert_array_equal(result[:, 0], captured[0][:, 3])
        self.assertFalse(np.allclose(result[:, 0], captured[0][:, 4]))
        self.assertFalse(np.allclose(result[:, 0], captured[0][:, 5]))

    def test_future_observations_do_not_change_current_state(self):
        changed = self.observed.copy()
        changed[32:] += 1000
        a = extract_vectors(self.model, self.observed, [31], [0, 1], 16, 1)
        b = extract_vectors(self.model, changed, [31], [0, 1], 16, 1)
        np.testing.assert_array_equal(a, b)

    def test_batching_does_not_mix_windows(self):
        alone = extract_vectors(self.model, self.observed, [31], [0, 1], 16, 1)
        together = extract_vectors(self.model, self.observed, [31, 95], [0, 1], 16, 2)
        np.testing.assert_allclose(alone[0], together[0], rtol=2e-6, atol=2e-6)

    def test_middle_block_convention(self):
        cfg = {"middle_layer_index": None, "capture_next_layer": True}
        self.assertEqual(resolve_layers(cfg, 12), [5, 6])


if __name__ == "__main__":
    unittest.main()
