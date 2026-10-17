import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Descriptive architecture controls, declared H2 model choice and specificity counterchecks")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "4"
    import numpy as np
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mosd_hidden_dynamics.study import (aggregate_results, depth_comparison, descriptive_statistics,
                                           load_group, write_rows)
    from mosd_hidden_dynamics.utils import save_json, source_record
    root = Path(args.run_dir).resolve()
    cfg = json.loads((root / "config_requested.json").read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    output = root / "research_controls"
    output.mkdir(exist_ok=True)
    conditions = ["pretrained"] + [f"random_{s}" for s in cfg["random_init_seeds"]]
    descriptions = []
    for condition in conditions:
        for dataset in cfg["datasets"]:
            for protocol in ("in_series", "out_of_series"):
                path = output / f"{condition}_{dataset['name']}_{protocol}_descriptive.json"
                if path.exists():
                    cached = json.loads(path.read_text())
                    if any(r["representation"] == "depth_transition_native" for r in cached):
                        descriptions.extend(cached)
                        continue
                group, _ = load_group(cfg, registry, root, dataset["name"], condition, protocol)
                values = descriptive_statistics(group, dataset.get("season_raw", 128))
                for split, part in group.items():
                    for series in np.unique(part["ids"]):
                        mask = part["ids"] == series
                        for kind, value in (("hidden_native", part["x"][mask]),
                                            ("observed_patch_native", part["raw"][mask, -16:])):
                            a, b = value[:-1], value[1:]
                            denominator = np.linalg.norm(a, axis=1).clip(1e-12)
                            values.append({"series_id": str(series), "split": split, "representation": kind,
                                "mean_relative_adjacent_jump": float(np.mean(np.linalg.norm(b - a, axis=1) / denominator)),
                                "lag1_cosine": float(np.mean(np.sum(a * b, axis=1) /
                                                           (denominator * np.linalg.norm(b, axis=1).clip(1e-12))))})
                        a, b = part["x"][mask], part["depth"][mask]
                        energy = np.square(b - a).mean(0)
                        ordered = np.sort(energy)
                        gini = float((2 * np.arange(1, len(ordered) + 1) - len(ordered) - 1) @ ordered
                                     / (len(ordered) * ordered.sum()))
                        norm_a, norm_b = np.linalg.norm(a, axis=1).clip(1e-12), np.linalg.norm(b, axis=1).clip(1e-12)
                        values.append({"series_id": str(series), "split": split, "representation": "depth_transition_native",
                            "mean_relative_adjacent_jump": float(np.mean(np.linalg.norm(b - a, axis=1) / norm_a)),
                            "lag1_cosine": float(np.mean(np.sum(a * b, axis=1) / (norm_a * norm_b))),
                            "jump_energy_gini": gini,
                            "acf_note": "Only blocks 5 and 6 are paired; depth ACF at lag 4 is not defined."})
                values = [{"dataset": dataset["name"], "condition": condition, "protocol": protocol, **v} for v in values]
                save_json(path, values)
                descriptions.extend(values)
                del group
                print("DESCRIBED", condition, dataset["name"], protocol, flush=True)
    save_json(output / "descriptive.json", descriptions)
    write_rows(output / "descriptive.csv", descriptions)
    rows = []
    for directory in (root / "remaining_baselines").iterdir():
        if directory.is_dir() and (directory / "complete.json").exists():
            rows.extend(json.loads((directory / "metrics.json").read_text()))
    waited = time.monotonic()
    while not (root / "control_mlp/complete.json").exists():
        if time.monotonic() - waited > 1800:
            raise TimeoutError("Noise MLP controls did not complete")
        time.sleep(2)
    for path in (root / "control_mlp").glob("*/metrics.json"):
        rows.extend(json.loads(path.read_text()))
    summary = aggregate_results(rows, cfg["bootstrap_samples"], cfg["seed"])
    save_json(output / "common_anchor_summary.json", summary)
    write_rows(output / "common_anchor_summary.csv", summary)
    selection = []
    for method in ("ridge_direct", "mlp_state"):
        gains = defaultdict(list)
        for row in summary:
            if row["model"] != method or row["horizon"] != 1:
                continue
            dataset = next(d for d in cfg["datasets"] if d["name"] == row["dataset"])
            if dataset["family"] == "control":
                continue
            for metric in ("r2", "cosine"):
                gains[(dataset["family"], row["protocol"], metric)].append(row[f"delta_{metric}"])
        means = [{"family": family, "protocol": protocol, "metric": metric, "gain": float(np.mean(value))}
                 for (family, protocol, metric), value in gains.items()]
        selection.append({"model": method, "minimum_gain": min(r["gain"] for r in means), "family_gains": means})
    selected = max(selection, key=lambda row: (row["minimum_gain"], row["model"] == "ridge_direct"))["model"]
    save_json(output / "primary_class_selection.json", {"criterion": "maximum minimum validation gain over persistence across metric/family/protocol",
              "aggregation": "seeds within series, series within dataset, datasets equally within family",
              "candidates": selection, "selected": selected, "test_evaluated": False})
    groups = defaultdict(list)
    for row in rows:
        if row["horizon"] == 1:
            groups[(row["dataset"], row["protocol"], row["model"], row["series_id"])].append(row)
    averaged = []
    for key, values in groups.items():
        averaged.append({"dataset": key[0], "protocol": key[1], "condition": "pretrained", "model": key[2],
                         "series_id": key[3], "horizon": 1, "seed": None,
                         **{metric: float(np.mean([v[metric] for v in values])) for metric in ("r2", "cosine")}})
    comparisons, specificity = [], []
    for temporal, depth in (("ridge_direct", "ridge_depth"), ("mlp_state", "mlp_depth")):
        pair = [{"ridge_direct": temporal, "ridge_depth": depth}, {"ridge_direct": "train_mean", "ridge_depth": depth}]
        for label, mapping in zip(("temporal_vs_depth", "constant_mean_vs_depth"), pair):
            reverse = {value: key for key, value in mapping.items()}
            converted = [{**r, "model": reverse[r["model"]]} for r in averaged if r["model"] in reverse]
            result = depth_comparison(converted, cfg["bootstrap_samples"], cfg["seed"])
            comparisons.extend({"comparison": label, "temporal_model_class": temporal, **r} for r in result)
        lookup = {(r["dataset"], r["protocol"], r["series_id"], r["model"]): r for r in averaged}
        for row in averaged:
            if row["model"] != temporal:
                continue
            key = (row["dataset"], row["protocol"], row["series_id"])
            mean = lookup[(*key, "train_mean")]
            specificity.append({"dataset": row["dataset"], "protocol": row["protocol"], "series_id": row["series_id"],
                                "model": temporal, **{f"gain_over_mean_{m}": row[m] - mean[m] for m in ("r2", "cosine")}})
    save_json(output / "declared_h2_and_counterchecks.json", comparisons)
    write_rows(output / "declared_h2_and_counterchecks.csv", comparisons)
    save_json(output / "specificity_over_mean.json", specificity)
    write_rows(output / "specificity_over_mean.csv", specificity)
    nulls = []
    for path in (root / "evaluations/main").glob("pretrained_*/null_controls.json"):
        values = json.loads(path.read_text())
        grouped = defaultdict(list)
        for value in values:
            grouped[(value["dataset"], value["protocol"], value["control"], value["seed"])].append(value)
        nulls.extend({"dataset": key[0], "protocol": key[1], "control": key[2], "seed": key[3],
                      **{metric: float(np.mean([v[metric] for v in trials])) for metric in ("r2", "cosine")}}
                     for key, trials in grouped.items())
    save_json(output / "null_seed_macro_scores.json", nulls)
    write_rows(output / "null_seed_macro_scores.csv", nulls)
    save_json(output / "run_metadata.json", {"source": source_record(repo), "test_evaluated": False,
              "interpretation": "Original H2 threshold preserved; mean/noise comparisons are specificity diagnostics, not a post-hoc replacement hypothesis.",
              "null_note": "Earlier common64/history1 anchor set; descriptive sanity distributions, not permutation-test p-values.",
              "mlp_note": "Same input scaler, capacity and optimization budget; output scalers fitted separately on each task's training targets."})
    save_json(output / "complete.json", {"descriptive_condition_groups": 40, "primary_class": selected, "test_evaluated": False})


if __name__ == "__main__":
    main()
