import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "4"

import numpy as np


DATASETS = ("Lorenz", "Rossler", "electricity")
MODELS = ("train_mean", "persistence", "ridge", "mlp", "residual", "ode")
SEEDS = (2026, 2027, 2028)
HORIZONS = (1, 4, 16, 64, 256)
METRICS = ("r2", "r2_standardized", "r2_whitened_pca95", "cosine", "centered_cosine", "relative_error")
LABELS = {"train_mean": "Training mean", "persistence": "Persistence", "ridge": "Ridge", "mlp": "MLP",
          "residual": "Residual", "ode": "Neural ODE", "chronos_median": "Chronos + re-encoding",
          "last_value": "Last value + re-encoding", "electricity": "Electricity", "white_noise": "White noise"}


def read(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def formatted(value):
    return f"{0.0 if abs(value) < .0005 else value:.3f}"


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for row in rows:
            saved = {}
            for key, value in row.items():
                if isinstance(value, (float, np.floating)):
                    precision_check = "relative_l2" in key or "relative_difference" in key
                    value = f"{value:.2e}" if precision_check and 0 < abs(value) < .0005 else formatted(value)
                saved[key] = value
            writer.writerow(saved)


def summarize(rows, keys, metrics=METRICS):
    groups, result = defaultdict(list), []
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    for key, group in sorted(groups.items()):
        ids = sorted({row["series_id"] for row in group})
        seeds = sorted({row.get("seed", 0) or 0 for row in group})
        row = dict(zip(keys, key)) | {"n_series": len(ids), "n_seeds": len(seeds)}
        samples = np.random.default_rng(2026).integers(len(ids), size=(2000, len(ids)))
        for metric in metrics:
            if all(r[metric] is None for r in group):
                row.update({metric: None, metric + "_ci_low": None, metric + "_ci_high": None, metric + "_seed_sd": None})
                continue
            assert all(r[metric] is not None for r in group), (key, metric)
            per_series = np.asarray([np.mean([r[metric] for r in group if r["series_id"] == sid]) for sid in ids])
            per_seed = [np.mean([r[metric] for r in group if (r.get("seed", 0) or 0) == seed]) for seed in seeds]
            assert np.isfinite(per_series).all()
            low, high = np.quantile(per_series[samples].mean(axis=1), [.025, .975])
            row.update({metric: float(per_series.mean()), metric + "_ci_low": float(low), metric + "_ci_high": float(high),
                        metric + "_seed_sd": float(np.std(per_seed, ddof=1)) if len(seeds) > 1 else 0.0})
        result.append(row)
    return result


def find(rows, dataset, model, horizon):
    found = [row for row in rows if row["dataset"] == dataset and row["model"] == model and row["horizon"] == horizon]
    assert len(found) == 1, (dataset, model, horizon, len(found))
    return found[0]


def markdown_table(rows, dataset, models, horizons, metric="r2", seed_sd=False):
    lines = ["| Model | " + " | ".join(f"k={k}" for k in horizons) + " |",
             "| --- | " + " | ".join("---:" for _ in horizons) + " |"]
    for model in models:
        values = []
        for horizon in horizons:
            row = find(rows, dataset, model, horizon)
            value = formatted(row[metric])
            if seed_sd and row["n_seeds"] > 1:
                value += " ± " + formatted(row[metric + "_seed_sd"])
            values.append(value)
        lines.append("| " + LABELS.get(model, model) + " | " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit and summarize the matched validation grid without refitting models")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mosd_hidden_dynamics.consistent import score_series
    root, output = Path(args.run_dir).resolve(), Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = read(root / "core/Lorenz/config_requested.json")
    assert cfg["seeds"] == list(SEEDS) and cfg["horizons"] == list(HORIZONS)
    sources, core, paired, paths, variance, resolution, correlations, oracle_checks = {}, [], [], [], [], [], [], []

    def load(path):
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read(path)

    def source_check(metadata):
        assert metadata["test_evaluated"] is False
        for name, digest in metadata["source"]["files"].items():
            assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest, name

    for dataset in DATASETS:
        directory = root / "core" / dataset
        assert load(directory / "complete.json")["status"] == "complete"
        assert load(directory / "config_requested.json") == cfg
        metadata = load(directory / "run_metadata.json")
        source_check(metadata)
        anchors = load(directory / "report_anchors.json")
        training_anchors = load(directory / "training_and_selection_anchors.json")
        assert {row["split"] for row in training_anchors} == {"train", "val"}
        assert {row["split"] for row in anchors} == {"val"}
        train_ids = {row["series_id"] for row in training_anchors if row["split"] == "train"}
        val_ids = {row["series_id"] for row in anchors}
        assert len(train_ids) == 8 and len(val_ids) == 4 and train_ids.isdisjoint(val_ids)
        with np.load(directory / "report_targets.npz") as saved:
            targets = dict(saved)
        np.testing.assert_array_equal(targets["ids"], [row["series_id"] for row in anchors])
        np.testing.assert_array_equal(targets["source_ends"], [row["source_end"] for row in anchors])
        with np.load(directory / "scoring_space.npz") as saved:
            space = dict(saved)
        for kind in ("coordinate", "pc"):
            values = np.sort(space[kind + "_variance"])[::-1]
            variance.extend({"dataset": dataset, "basis": kind, "top": count,
                             "variance_fraction": float(values[:count].sum() / values.sum()),
                             "pca95_rank": int(space["rank"])} for count in (1, 5, 20))
        plugin_dir = root / "reencoding" / dataset
        plugin_meta = load(plugin_dir / "run_metadata.json")
        source_check(plugin_meta)
        assert plugin_meta["report_anchor_sha256"] == metadata["report_anchor_sha256"]
        selected_anchors = load(plugin_dir / "selected_anchors.json")
        with np.load(plugin_dir / "pretrained_predictions.npz") as saved:
            selected = saved["selected_indices"]
            np.testing.assert_array_equal(targets["ids"][selected], saved["ids"])
            assert selected_anchors == [anchors[index] for index in selected]
            for horizon in cfg["plugin_horizons"]:
                np.testing.assert_array_equal(targets[f"y{horizon}"][selected], saved[f"target{horizon}"])
        for model in MODELS:
            for seed in SEEDS if model in ("mlp", "residual", "ode") else (0,):
                prefix = directory / f"{model}_{seed}"
                if seed:
                    assert load(prefix.with_name(prefix.name + "_complete.json"))["status"] == "complete"
                    training = load(prefix.with_name(prefix.name + "_training.json"))
                    assert 1 <= training["best_epoch"] <= cfg["training"]["epochs"]
                rows = load(prefix.with_name(prefix.name + "_metrics.json"))
                expected = {(sid, horizon) for sid in val_ids for horizon in HORIZONS}
                assert len(rows) == len(expected)
                assert {(row["series_id"], row["horizon"]) for row in rows} == expected
                assert all(row["status"] == "complete" for row in rows)
                core.extend(rows)
                with np.load(prefix.with_name(prefix.name + "_predictions.npz")) as predictions:
                    for horizon in HORIZONS:
                        assert predictions[f"k{horizon}"].shape == targets[f"y{horizon}"].shape
                        assert np.isfinite(predictions[f"k{horizon}"]).all()
                        if horizon in cfg["plugin_horizons"]:
                            matched = score_series(targets[f"y{horizon}"][selected], predictions[f"k{horizon}"][selected],
                                targets[f"reference{horizon}"], targets["ids"][selected], space)
                            paired.extend({"dataset": dataset, "model": model, "seed": seed, "horizon": horizon, **row}
                                          for row in matched)
                path = load(prefix.with_name(prefix.name + "_path_metrics.json"))
                assert "status" not in path
                paths.append({"dataset": dataset, "model": model, "seed": seed, **path})
                if model == "ode":
                    value = load(prefix.with_name(prefix.name + "_resolution.json"))
                    resolution.append({"dataset": dataset, "seed": seed,
                        "relative_difference_8_vs_16_k4": value["relative_prediction_difference_8_vs_16_k4"],
                        "macro_r2_substeps16_k4_selection_anchors": float(np.mean([r["r2"] for r in value["substeps16_entity_metrics"]]))})
        print("Audited core and paired anchors:", dataset, flush=True)

    plugin_rows = []
    for dataset in DATASETS + ("white_noise",):
        directory = root / "reencoding" / dataset
        source_check(load(directory / "run_metadata.json"))
        for condition in ("pretrained",) + tuple(f"random_{seed}" for seed in SEEDS):
            assert load(directory / f"{condition}_complete.json")["status"] == "complete"
            rows = load(directory / f"{condition}_metrics.json")
            assert len(rows) == (2 if dataset == "white_noise" else 4) * 4 * 3
            plugin_rows.extend({**row, "encoder": "random" if condition.startswith("random") else "pretrained",
                                "model": row["method"]} for row in rows)
            checks = load(directory / f"{condition}_oracle_checks.json")
            assert max(row["max_relative_l2"] for row in checks) < .002
            oracle_checks.extend({"dataset": dataset, "condition": condition, **row} for row in checks)
            errors = load(directory / f"{condition}_errors.json")
            for horizon in cfg["plugin_horizons"]:
                for model in ("chronos_median", "last_value"):
                    selected = [row for row in errors if row["horizon"] == horizon and row["method"] == model]
                    x, y = np.asarray([[r["raw_window_nrmse"], r["latent_relative_error"]] for r in selected]).T
                    correlations.append({"dataset": dataset, "condition": condition, "model": model, "horizon": horizon,
                        "n_anchors": len(selected), "pearson_raw_vs_latent_error": float(np.corrcoef(x, y)[0, 1]),
                        "mean_raw_window_nrmse": float(x.mean()), "mean_latent_relative_error": float(y.mean())})
        paired.extend({**row, "seed": 0, "model": row["method"]} for row in
                      load(directory / "pretrained_metrics.json") if row["method"] != "persistence")
    null_meta = load(root / "reencoding/white_noise/null_ridge_metadata.json")
    source_check(null_meta)
    assert null_meta["report_anchor_sha256"] == load(root / "reencoding/white_noise/run_metadata.json")["report_anchor_sha256"]
    paired.extend({**row, "model": "ridge", "seed": 0} for row in load(root / "reencoding/white_noise/null_ridge_metrics.json"))
    control = load(root / "positive_control/result.json")
    source_check(control)
    assert control["status"] == "passed" and control["validation_metrics"]["r2"] >= .95
    assert control["relative_difference_8_vs_16"] < .001
    core_summary = summarize(core, ["dataset", "model", "horizon"])
    paired_summary = summarize(paired, ["dataset", "model", "horizon"])
    plugin_summary = summarize(plugin_rows, ["dataset", "encoder", "model", "horizon"])
    differences = []
    for dataset in DATASETS + ("white_noise",):
        for horizon in cfg["plugin_horizons"]:
            comparator = [row for row in paired if row["dataset"] == dataset and row["model"] == "chronos_median" and row["horizon"] == horizon]
            for model in (("ridge",) if dataset == "white_noise" else ("ridge", "mlp", "residual", "ode")):
                for reference in comparator:
                    values = [row for row in paired if row["dataset"] == dataset and row["model"] == model
                              and row["horizon"] == horizon and row["series_id"] == reference["series_id"]]
                    for value in values:
                        differences.append({"dataset": dataset, "model": model, "horizon": horizon,
                            "seed": value["seed"], "series_id": value["series_id"],
                            **{metric: value[metric] - reference[metric] for metric in METRICS}})
    delta_summary = summarize(differences, ["dataset", "model", "horizon"])
    path_summary = []
    path_metrics = ("sliced_w2_pc8_tail128", "acf_mae_pc2", "psd_relative_l2_pc2", "bounded_fraction", "constant_projected_trajectory_fraction")
    for dataset in DATASETS:
        for model in MODELS:
            values = [row for row in paths if row["dataset"] == dataset and row["model"] == model]
            path_summary.append({"dataset": dataset, "model": model, "n_seeds": len(values),
                                 **{metric: float(np.mean([r[metric] for r in values])) for metric in path_metrics}})
    for name, rows in (("core_summary", core_summary), ("core_entities", core), ("paired_summary", paired_summary),
                       ("paired_entities", paired), ("paired_gain_over_reencoding", delta_summary), ("encoder_controls", plugin_summary),
                       ("path_statistics", path_summary), ("variance_concentration", variance), ("ode_resolution", resolution),
                       ("input_latent_error", correlations), ("oracle_precision", oracle_checks)):
        write_csv(output / (name + ".csv"), rows)
    lines = ["# Matched hidden-state transition comparison", "", "Validation results, 26 October 2026.", "",
             "Frozen Chronos-2 block 5; context 512; one analysis step advances 16 observations. "
             "Each dataset has eight training and four unseen validation series. All horizons and models use the same reporting starts. "
             "Every transition is applied recursively without true future states.", "",
             "Ridge fits one-step pairs, with regularization selected by recursive four-step validation R². "
             "MLP, residual and ODE share a 768 → 256 → 768 Tanh network and a four-step training loss, "
             "with three initialization seeds. ODE uses RK4 with four substeps during training and rollout. "
             "The backbone is never fine-tuned.", "",
             "## Main comparison", "", "Macro R² uses each horizon's training-target mean as its reference. "
             "Zero equals that constant baseline; negative values are worse. Values after ± are sample standard deviations "
             "of the three seed-level macro scores, not confidence intervals.", ""]
    for dataset in DATASETS:
        lines += ["### " + LABELS.get(dataset, dataset), "", markdown_table(core_summary, dataset, MODELS, HORIZONS, seed_sd=True), ""]
    lines += ["![Recursive state prediction](horizon_comparison.png)", "",
              "Figure 1. Mean R² versus recursive horizon, evaluated on identical starts within each dataset. "
              "Shading shows ± one seed standard deviation for neural models. The vertical axis uses a symmetric logarithmic scale "
              "to show strongly negative long-horizon scores; numeric values are in the tables. The dashed line is the training-mean baseline.", "",
              "## Forecast the input, then encode it", "", "This separate comparison uses 32 fixed starts per validation series, "
              "a subset of the main report manifest. All methods below are rescored on exactly those starts. "
              "Chronos predicts the median continuation from the past window; the predicted rolling window is then re-encoded. "
              "Last-value continuation is a simpler input baseline. It is not valid to compare this subset directly with the full-grid table above.", ""]
    for dataset in DATASETS:
        lines += ["### " + LABELS.get(dataset, dataset), "", markdown_table(paired_summary, dataset,
                  ("ridge", "mlp", "residual", "ode", "chronos_median", "last_value"), cfg["plugin_horizons"]), ""]
    lines += ["### White-noise specificity check", "", markdown_table(paired_summary, "white_noise",
              ("ridge", "chronos_median", "last_value"), cfg["plugin_horizons"]), "",
              "A gain over input re-encoding also occurs on white noise. It therefore cannot establish extra predictive information, "
              "a meaningful autonomous law, or the original H2 by itself.", "",
              "![Matched input re-encoding controls](reencoding_comparison.png)", "",
              "Figure 2. R² on the matched re-encoding subset. The curves compare recursive hidden-state readouts with two "
              "input-continuation baselines. These are different prediction objectives and do not establish a causal explanation for readout gains.", "",
              "## How to read the supporting files", "",
              "- `core_summary.csv`: full-grid scores, seed variation and 95% percentile intervals resampling validation series after averaging seeds within each series.",
              "- `paired_summary.csv` and `paired_gain_over_reencoding.csv`: matched-subset scores and paired readout-minus-Chronos-re-encoding differences, with series-level intervals.",
              "- `core_entities.csv` and `paired_entities.csv`: separate results for every validation series and seed.",
              "- `encoder_controls.csv`: re-encoding scores in pretrained and three randomly initialized encoder spaces. Each condition is evaluated against its own target; cross-encoder R² differences do not identify a pretraining effect.",
              "- `variance_concentration.csv`: training variance in the largest 1, 5 and 20 coordinates or principal components; `core_summary.csv` also reports coordinate-standardized and whitened PCA95 R².",
              "- `path_statistics.csv`: 256-step finite-rollout diagnostics on the same 16 starts per series. Sliced Wasserstein distance uses the last 128 states in eight whitened training PCs; ACF error uses lags 1, 4, 16 and 64, and PSD error uses the first two PCs. Smaller distances/errors are better.",
              "- `bounded_fraction` requires every predicted state to stay within ten times the training 99th-percentile centered radius. It is a coarse explosion check, not a physical stability theorem. Constant paths can pass it.",
              "- `ode_resolution.csv`: eight-versus-sixteen-substep prediction differences at k=4, on the original selection anchors; these scores must not be mixed with the long report manifest.",
              "- `oracle_precision.csv`: encoding true future windows agrees with cached FP16 states within the stated 0.2% relative-L2 tolerance. This is an implementation check, not a forecast.",
              "- `input_latent_error.csv`: descriptive raw-window/latent-error correlations across overlapping anchors, without significance claims.", "",
              "## Interpretation limits", "", "There are only four independent validation series per main dataset and two for white noise. "
              "Bootstrap intervals are conditional on this small set; seeds and overlapping windows are not independent new series. "
              "The same validation entities are used for model selection and reporting, so these are development results, not unbiased final-test estimates. "
              "For synthetic systems, unseen series change initial conditions at fixed parameters. The finite trajectory statistics do not prove recovery "
              "of an invariant measure or a chaotic attractor. The original H1/H2 are unchanged and no final test is evaluated.", "",
              "The ODE implementation control passes on a known stable oscillator (four-step R² above 0.999). "
              "This verifies one implementation/training case and does not establish ODE suitability for the hidden states.", "",
              "Reproduction: run `scripts/build_consistent_report.py --run-dir <saved-run> --output-dir <report-directory>` "
              "after the grid, input re-encoding and white-noise control complete. Raw features, checkpoints and logs remain outside the repository."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.dpi": 180})
    colors = {"ridge": "#2274a5", "mlp": "#e08723", "residual": "#7c589f", "ode": "#ba4040",
              "persistence": "#777777", "chronos_median": "#28845d", "last_value": "#777777"}
    for filename, summary, methods, horizons in (
            ("horizon_comparison", core_summary, ("ridge", "mlp", "residual", "ode", "persistence"), HORIZONS),
            ("reencoding_comparison", paired_summary, ("ridge", "mlp", "chronos_median", "last_value"), cfg["plugin_horizons"])):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for axis, dataset in zip(axes, DATASETS):
            for model in methods:
                selected = [find(summary, dataset, model, horizon) for horizon in horizons]
                y = np.asarray([row["r2"] for row in selected])
                axis.plot(horizons, y, marker="o", markersize=4, color=colors[model], label=LABELS[model])
                if model in ("mlp", "residual", "ode"):
                    sd = np.asarray([row["r2_seed_sd"] for row in selected])
                    axis.fill_between(horizons, y - sd, y + sd, color=colors[model], alpha=.12)
            axis.axhline(0, color="black", linestyle="--", linewidth=.8)
            axis.set_xscale("log", base=4)
            axis.set_xticks(horizons, [str(k) for k in horizons])
            axis.set_yscale("symlog", linthresh=1)
            axis.set_ylim(top=1.1)
            axis.set_title(LABELS.get(dataset, dataset))
            axis.set_xlabel("Recursive horizon k (1 step = 16 observations)")
            axis.grid(alpha=.2)
        axes[0].set_ylabel("R² relative to training mean (symlog scale)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(methods), frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, .12, 1, 1))
        fig.savefig(output / (filename + ".png"))
        fig.savefig(output / (filename + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, dataset in zip(axes, DATASETS):
        directory = root / "core" / dataset
        for filename, label, color in (("observed_paths", "Observed", "#333333"), ("ridge_0_paths", "Ridge", "#2274a5"),
                                       ("mlp_2026_paths", "MLP", "#e08723")):
            with np.load(directory / (filename + ".npz")) as data:
                values = data["paths"][0]
            axis.plot(values[:, 0], values[:, 1], color=color, linewidth=1, alpha=.8, label=label)
        axis.scatter(values[0, 0], values[0, 1], s=20, color="black", zorder=5, label="Initial state")
        axis.set_title(LABELS.get(dataset, dataset))
        axis.set_xlabel("Training PC1")
        axis.set_ylabel("Training PC2")
        axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(output / "projected_paths.png")
    fig.savefig(output / "projected_paths.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    audit = {"status": "passed", "test_evaluated": False, "datasets": list(DATASETS), "neural_fits": 27,
             "core_entity_rows": len(core), "paired_entity_rows": len(paired), "inputs_sha256": sources,
             "positive_control": {"status": control["status"], "r2": control["validation_metrics"]["r2"]},
             "oracle_max_relative_l2": max(row["max_relative_l2"] for row in oracle_checks),
             "ode_max_relative_difference_8_vs_16_k4": max(row["relative_difference_8_vs_16_k4"] for row in resolution),
             "checks": ["all requested fits and horizons complete", "finite saved predictions", "disjoint train and validation entities",
                        "shared report starts across horizons and models", "re-encoding targets exactly match selected core targets",
                        "source files match recorded fingerprints", "known-flow control passes", "oracle cache precision within tolerance"],
             "limits": "No independent final-test evaluation; statistics conditional on observed validation entities."}
    write_json(root / "audit_consistent.json", audit)
    print(json.dumps({key: value for key, value in audit.items() if key != "inputs_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
