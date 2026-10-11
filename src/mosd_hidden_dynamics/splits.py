import csv
import hashlib

import numpy as np


def make_manifest(n_raw, context, stride, horizon, train_fraction=0.7, val_end_fraction=0.8):
    if not 0 < train_fraction < val_end_fraction < 1:
        raise ValueError("Invalid chronological split fractions")
    if context <= 0 or stride <= 0 or horizon <= 0 or context % stride:
        raise ValueError("Context must be a positive multiple of stride; horizon must be positive")
    train_end, val_end = int(n_raw * train_fraction), int(n_raw * val_end_fraction)
    bounds = {
        "train": (0, train_end),
        "val": (train_end + context, val_end),
        "test": (val_end + context, n_raw),
    }
    rows = []
    for index, end in enumerate(range(context - 1, n_raw - horizon * stride, stride)):
        split = "train" if end < train_end else "val" if end < val_end else "test"
        start, stop = bounds[split]
        target_end = end + horizon * stride
        eligible = end - context + 1 >= start and target_end < stop
        rows.append({
            "series_id": "lorenz_0", "analysis_index": index,
            "raw_context_start": end - context + 1, "raw_context_end": end,
            "target_analysis_index": index + horizon,
            "target_context_start": target_end - context + 1, "target_context_end": target_end,
            "context_length_raw": context, "analysis_stride_raw": stride,
            "horizon_patches": horizon, "split": split, "eligible": eligible,
            "exclusion_reason": "" if eligible else "boundary_or_embargo",
        })
    return rows, bounds


def audit_manifest(rows, bounds, observed):
    if len({r["analysis_index"] for r in rows}) != len(rows):
        raise ValueError("Duplicate analysis indices")
    counts, hashes, receptive_bounds = {}, {}, {}
    for split in bounds:
        selected = [r for r in rows if r["split"] == split and r["eligible"]]
        if len(selected) < 2:
            raise ValueError(f"Too few eligible {split} pairs after embargo: {len(selected)}")
        counts[split] = len(selected)
        split_hashes = set()
        for r in selected:
            start, stop = bounds[split]
            if not start <= r["raw_context_start"] <= r["target_context_end"] < stop:
                raise ValueError("A pair crosses its eligible raw-data interval")
            if r["target_context_end"] - r["raw_context_end"] != r["analysis_stride_raw"] * r["horizon_patches"]:
                raise ValueError("Incorrect target alignment")
            for a, b in ((r["raw_context_start"], r["raw_context_end"]),
                         (r["target_context_start"], r["target_context_end"])):
                split_hashes.add(hashlib.sha256(observed[a:b + 1].tobytes()).hexdigest())
        hashes[split] = split_hashes
        receptive_bounds[split] = (
            min(r["raw_context_start"] for r in selected),
            max(r["target_context_end"] for r in selected),
        )
    gaps = {}
    for a, b in (("train", "val"), ("val", "test")):
        gap = receptive_bounds[b][0] - receptive_bounds[a][1] - 1
        if gap < rows[0]["context_length_raw"]:
            raise ValueError("Receptive fields violate the full-context embargo")
        if hashes[a] & hashes[b]:
            raise ValueError("Exact duplicate raw windows across splits")
        gaps[f"{a}_{b}"] = gap
    if hashes["train"] & hashes["test"]:
        raise ValueError("Exact duplicate train/test windows")
    return {"passed": True, "eligible_pairs": counts, "raw_bounds_half_open": bounds,
            "receptive_field_gaps_raw": gaps, "exact_cross_split_duplicates": 0,
            "test_metrics_evaluated": False}


def save_manifest(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in row:
            if key not in ("series_id", "split", "eligible", "exclusion_reason"):
                row[key] = int(row[key])
        row["eligible"] = row["eligible"] == "True"
    return rows
