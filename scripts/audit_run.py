import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mosd_hidden_dynamics.splits import audit_manifest, load_manifest, make_manifest
from mosd_hidden_dynamics.utils import sha256_file


def main():
    parser = argparse.ArgumentParser(description="Recheck a recorded run's data, manifest and extraction membership")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir)
    cfg = json.loads((root / "config_requested.json").read_text())
    metadata = json.loads((root / "run_metadata.json").read_text())
    model_config = json.loads((root / "model_config.json").read_text())
    if sha256_file(root / "data.npz") != metadata["data"]["file_sha256"]:
        raise ValueError("Prepared data checksum mismatch")
    if sha256_file(root / "split_manifest.csv") != metadata["split_manifest_sha256"]:
        raise ValueError("Split checksum mismatch")
    with np.load(root / "data.npz") as data:
        observed = data["observed"]
    rows = load_manifest(root / "split_manifest.csv")
    expected, bounds = make_manifest(
        len(observed), cfg["state_extraction"]["context_length_raw"],
        model_config["chronos_config"]["input_patch_stride"],
        cfg["prediction"]["horizon_patches"], **cfg["split"],
    )
    if rows != expected:
        raise ValueError("Manifest does not match its recorded generation settings")
    audit = audit_manifest(rows, bounds, observed)
    if (root / "states.npz").exists():
        extraction = json.loads((root / "extraction_metadata.json").read_text())
        if sha256_file(root / "states.npz") != extraction["file_sha256"]:
            raise ValueError("Activation checksum mismatch")
        expected_endpoints = sorted({r[key] for r in rows if r["eligible"] and r["split"] in ("train", "val")
                                     for key in ("raw_context_end", "target_context_end")})
        with np.load(root / "states.npz") as states:
            np.testing.assert_array_equal(states["raw_endpoints"], expected_endpoints)
            if not np.isfinite(states["states"]).all():
                raise ValueError("Non-finite activations")
            audit["extracted_states"] = len(states["raw_endpoints"])
        if extraction["test_states_extracted"] or extraction["future_targets_or_covariates_supplied"]:
            raise ValueError("Extraction violates the checkpoint policy")
    if (root / "metrics.json").exists():
        metrics = json.loads((root / "metrics.json").read_text())
        if metrics["test_evaluated"] or metrics["split"] != "validation":
            raise ValueError("Metrics violate the validation-only policy")
        if metrics["split_manifest_sha256"] != metadata["split_manifest_sha256"]:
            raise ValueError("Metrics refer to a different split manifest")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
