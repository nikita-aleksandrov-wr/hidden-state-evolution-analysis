import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASETS = ["Lorenz", "Rossler", "electricity"]
LABELS = {"electricity": "Electricity", "electricity_extra": "Extra meters", "solar": "Solar"}


def formatted(value):
    return f"{0.0 if abs(value) < .0005 else value:.3f}"


def read(path):
    return json.loads(path.read_text())


def mean_rows(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)
    result = []
    for key, values in groups.items():
        per_series = defaultdict(list)
        for row in values:
            per_series[row["series_id"]].append(row)
        scores = {m: float(np.mean([np.mean([r[m] for r in trials]) for trials in per_series.values()]))
                  for m in metrics}
        seeds = {r["seed"] for r in values if r.get("seed") is not None}
        spread = {m + "_std_over_seeds": float(np.std([
            np.mean([r[m] for r in values if r.get("seed") == seed]) for seed in seeds]))
            for m in metrics} if len(seeds) > 1 else {}
        result.append(dict(zip(keys, key)) | scores | spread |
                      {"n_series": len(per_series), "n_seeds": len(seeds) if seeds else 1})
    return result


def lookup(rows, dataset, model, horizon=1, protocol="out_of_series", **filters):
    found = [r for r in rows if r["dataset"] == dataset and r["model"] == model
             and r.get("horizon", horizon) == horizon and r["protocol"] == protocol
             and all(r.get(k) == v for k, v in filters.items())]
    if len(found) != 1:
        raise ValueError((dataset, model, horizon, filters, len(found)))
    return found[0]


def table(path, columns, rows):
    lines = [r"\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}", r"\toprule",
             " & ".join(columns) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(formatted(v) if isinstance(v, (float, np.floating)) else str(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build rounded review tables and scientific figures from saved validation outputs")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--additional-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paper-data-dir", required=True)
    args = parser.parse_args()
    root, extra = Path(args.run_dir), Path(args.additional_run)
    output, paper = Path(args.output_dir), Path(args.paper_data_dir)
    output.mkdir(parents=True, exist_ok=True)
    paper.mkdir(parents=True, exist_ok=True)
    assert read(root / "audit_review.json")["status"] == "passed"
    sources, all_rows = {}, []
    def load(section, path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = read(path)
        all_rows.extend({"section": section, **r} for r in rows)
        return rows
    baseline = load("common_64", root / "remaining_baselines/summary.json")
    ode = load("ode_64", root / "neural_ode/summary.json")
    long = load("common_256", root / "ode_full_rollouts/summary.json")
    resolution = mean_rows(read(root / "ode_resolution/metrics.json"),
        ["dataset", "protocol", "model", "section", "substeps", "horizon", "subset"],
        ["r2", "cosine", "relative_error"])
    all_rows.extend({**r, "section": "ode_resolution_" + r["section"]} for r in resolution)
    residual = load("residual_256", root / "residual_training/summary.json")
    ablations = load("ablations", root / "ablations/summary.json")
    additional = load("additional", extra / "evaluations/main/summary.json")
    transfer = mean_rows(read(extra / "frozen_transfer/metrics.json"),
                         ["dataset", "source_dataset", "protocol", "model", "horizon"], ["r2", "cosine"])
    all_rows.extend({"section": "frozen_transfer", **r} for r in transfer)
    dense_path = root / "ode_interpolation/metrics.json"
    dense = mean_rows(read(dense_path), ["dataset", "protocol", "model", "subset"], ["relative_error", "r2", "cosine"])
    all_rows.extend({"section": "interpolation", "horizon": 4, **r} for r in dense)
    selection = read(root / "research_controls/primary_class_selection.json")
    h2 = read(root / "research_controls/declared_h2_and_counterchecks.json")
    all_rows.extend({"section": "h2_specificity", "model": r["temporal_model_class"], "horizon": 1, **r} for r in h2)
    raw_descriptions = read(root / "research_controls/descriptive.json")
    desc = mean_rows([r for r in raw_descriptions
                      if r["split"] == "val" and r["representation"] in ("hidden_native", "observed_patch_native")],
                     ["dataset", "condition", "protocol", "representation"],
                     ["mean_relative_adjacent_jump", "lag1_cosine"])
    desc.extend(mean_rows([r for r in raw_descriptions
                           if r["split"] == "val" and r["representation"] == "depth_transition_native"],
                          ["dataset", "condition", "protocol", "representation"],
                          ["mean_relative_adjacent_jump", "lag1_cosine", "jump_energy_gini"]))
    all_rows.extend({"section": "native_description", "model": r["representation"], **r} for r in desc)
    standardized = []
    for row in read(root / "research_controls/descriptive.json"):
        if row["split"] == "val" and row["representation"] in ("hidden", "observed_patch"):
            seasonal_key = next(k for k in row if k.startswith("acf_lag_") and k not in ("acf_lag_1", "acf_lag_4"))
            standardized.append({**row, "acf_reference_lag": row[seasonal_key]})
    standardized = mean_rows(standardized, ["dataset", "condition", "protocol", "representation"],
        ["relative_jump_rms", "jump_energy_gini", "acf_lag_1", "acf_lag_4", "acf_reference_lag"])
    all_rows.extend({"section": "scaled_description", "model": r["representation"], **r} for r in standardized)
    columns = ["section", "dataset", "source_dataset", "protocol", "condition", "model", "horizon", "subset",
               "r2", "cosine", "relative_error", "n_series", "n_seeds", "comparison", "substeps",
               "relative_jump_rms", "jump_energy_gini", "acf_lag_1", "acf_lag_4", "acf_reference_lag",
               "r2_std_over_seeds", "cosine_std_over_seeds",
               "delta_r2", "delta_cosine", "mean_relative_adjacent_jump", "lag1_cosine",
               "delta_r2_ci_low", "delta_r2_ci_high", "delta_cosine_ci_low", "delta_cosine_ci_high"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: formatted(v) if isinstance(v, float) else v for k, v in row.items()})
    short = []
    for dataset in DATASETS:
        for k in (1, 4):
            short.append([LABELS.get(dataset, dataset), k] +
                         [lookup(baseline, dataset, m, k)["r2"] for m in
                          ("persistence", "ridge_direct", "mlp_state", "mlp_history", "ridge_observed")] +
                         [lookup(ode, dataset, "neural_ode", k)["r2"]])
    table(paper / "short_results.tex",
          ["Dataset", "$k$", "Persist.", "Ridge", "MLP", "MLP+history", "Ridge+obs.", "ODE"], short)
    table(paper / "additional_results.tex", ["Dataset", "Ridge 1", "MLP 1", "Ridge 4", "MLP 4"],
          [[LABELS.get(d, d)] + [lookup(additional, d, m, k)["r2"] for k in (1, 4)
                                for m in ("ridge_direct", "mlp_direct")]
           for d in ("Thomas", "Aizawa", "solar", "electricity_extra")])
    interpolation = []
    for dataset in DATASETS:
        interpolation.append([LABELS.get(dataset, dataset)] +
            [lookup(dense, dataset, method, subset="all_future")["relative_error"]
             for method in ("ridge_recursive_linear_bridge", "neural_ode")] +
            [lookup(resolution, dataset, "neural_ode", 4, section="dense", substeps=16,
                    subset="all_future")["relative_error"]])
    table(paper / "interpolation_results.tex",
          ["Dataset", "Ridge", "ODE ($1/4$)", "ODE ($1/16$)"], interpolation)
    table(paper / "long_results.tex", ["Dataset", "Ridge", "ODE (1)", "ODE ($1/16$)"],
          [[LABELS.get(d, d), lookup(long, d, "ridge_original", 256)["r2"], lookup(long, d, "neural_ode", 256)["r2"],
            lookup(resolution, d, "neural_ode", 256, section="long", substeps=16)["r2"]]
           for d in DATASETS])
    table(paper / "reduced_results.tex", ["Dataset", "PCs", "DMD $k=1$", "VAR(8) $k=1$", "DMD $k=4$", "VAR(8) $k=4$"],
          [[LABELS.get(d, d), read(root / "remaining_baselines" / f"{d}_out_of_series/pca.json")["rank"]] +
           [lookup(baseline, d, method, k)["r2"] for k in (1, 4) for method in ("dmd_recursive", "var_p8")]
           for d in DATASETS])
    native = []
    for d in DATASETS:
        values = [next(r for r in desc if r["dataset"] == d and r["protocol"] == "out_of_series"
                       and r["condition"] == "pretrained" and r["representation"] == rep)
                  for rep in ("hidden_native", "observed_patch_native")]
        native.append([LABELS.get(d, d)] + [r["mean_relative_adjacent_jump"] for r in values]
                      + [r["lag1_cosine"] for r in values])
    table(paper / "descriptive_results.tex", ["Dataset", "Jump $h$", "Jump $x$", "Cos. $h$", "Cos. $x$"], native)
    table(paper / "h2_results.tex", ["Input", r"$\Delta R^2$", r"$\Delta$cos.", "Mean beats depth"],
          [[LABELS.get(d, d).replace("white_noise", "White noise")] +
            [next(r for r in h2 if r["dataset"] == d and r["protocol"] == "out_of_series"
                  and r["temporal_model_class"] == selection["selected"] and r["comparison"] == "temporal_vs_depth")[metric]
             for metric in ("delta_r2", "delta_cosine")] +
            ["Yes" if next(r for r in h2 if r["dataset"] == d and r["protocol"] == "out_of_series"
                          and r["temporal_model_class"] == selection["selected"]
                          and r["comparison"] == "constant_mean_vs_depth")["local_threshold_met"] else "No"]
            for d in DATASETS + ["white_noise"]])
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 190, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), sharey=True)
    methods = [("ridge_original", "Ridge", long), ("neural_ode", "ODE dt=1", long),
               ("residual_training1", "Residual (1)", residual),
               ("residual_training8", "Residual (8)", residual),
               ("ridge_radius_0.99", r"Ridge $\rho\leq0.99$", residual)]
    fine_long = [{**r, "model": "neural_ode_fine"} for r in resolution
                 if r["section"] == "long" and r["substeps"] == 16]
    methods.append(("neural_ode_fine", "ODE dt=1/16", fine_long))
    for ax, d in zip(axes, DATASETS):
        for method, label, rows in methods:
            values = sorted([r for r in rows if r["dataset"] == d and r["model"] == method], key=lambda r: r["horizon"])
            ax.plot([r["horizon"] for r in values], [1 - r["r2"] for r in values], "o-", ms=3, lw=1.3, label=label)
        ax.axhline(1, color=".5", ls="--", lw=.8)
        ax.set(xscale="log", yscale="log", xlabel="Analysis steps (16 observations each)", title=LABELS.get(d, d))
        ax.grid(alpha=.18)
    axes[0].set_ylabel(r"$1-\mathrm{macro}\ R^2$ (lower is better)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .2, 1, 1))
    fig.savefig(output / "long_rollouts.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for d in DATASETS + ["white_noise"]:
        ax = axes[0]
        values = [lookup(ablations, d, "ridge_direct", condition=f"context_{c}_block5_stride16_calibrationFalse")["r2"]
                  for c in (16, 128, 512, 1024)]
        ax.plot([16, 128, 512, 1024], values, "o-", label=LABELS.get(d, d).replace("white_noise", "White noise"))
        ax = axes[1]
        values = [lookup(ablations, d, "ridge_direct", condition=f"context_512_{layer}_stride16_calibrationFalse")["r2"]
                  for layer in ("block0", "block5", "block11", "final_norm")]
        ax.plot(range(4), values, "o-")
    axes[0].set(xscale="log", xlabel="Context observations", ylabel=r"One-step validation $R^2$", title="Context (block 5)")
    axes[1].set(xticks=range(4), xticklabels=["Block 0", "Block 5", "Block 11", "Final norm"],
                title="Layer (context 512)", ylabel=r"One-step validation $R^2$")
    for ax in axes:
        ax.grid(alpha=.18)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(output / "controls.png")
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.7))
    for col, d in enumerate(DATASETS):
        with np.load(root / "ode_full_rollouts" / d / "trajectories.npz") as data:
            ids = data["series_ids"]
            chosen = [int(np.flatnonzero(ids == sid)[0]) for sid in np.unique(ids)[:3]]
            for row, stop in enumerate((5, 257)):
                for key, label, colour in (("observed", "Observed", "#263238"),
                                           ("ridge_original", "Ridge", "#2678b3"),
                                           ("neural_ode_seed2026", "ODE", "#d55e00")):
                    for j, index in enumerate(chosen):
                        trajectory = data[key][index, :stop]
                        axes[row, col].plot(trajectory[:, 0], trajectory[:, 1], color=colour,
                                            lw=1, alpha=.8, label=label if j == 0 else None)
                        axes[row, col].scatter(*trajectory[0], color=colour, s=9)
                axes[row, col].set(xlabel="PC1", ylabel="PC2",
                    title=LABELS.get(d, d) + (" (0–4 steps)" if row == 0 else " (0–256 steps)"))
                axes[row, col].grid(alpha=.15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .055, 1, 1))
    fig.savefig(output / "horizons.png")
    plt.close(fig)
    from matplotlib.colors import LogNorm
    targets = ["Thomas", "Aizawa", "solar", "electricity_extra"]
    matrix = np.array([[lookup(transfer, d, "ridge_direct", source_dataset=s)["r2"] for s in DATASETS] for d in targets])
    fig, ax = plt.subplots(figsize=(6.7, 3.1))
    norm = LogNorm(vmin=max(1e-4, float((1 - matrix).min())), vmax=float((1 - matrix).max()))
    chart = ax.imshow(1 - matrix, cmap="viridis_r", norm=norm, aspect="auto")
    for row in range(4):
        for col in range(3):
            ax.text(col, row, f"{matrix[row, col]:.3f}", ha="center", va="center",
                    color="white" if norm(1 - matrix[row, col]) > .52 else "black")
    ax.set(xticks=range(3), xticklabels=[LABELS.get(d, d) for d in DATASETS],
           yticks=range(4), yticklabels=[LABELS.get(d, d) for d in targets],
           xlabel="Frozen source map", ylabel="Unseen target dataset", title=r"One-step cross-dataset validation $R^2$")
    fig.colorbar(chart, ax=ax, label=r"Color: $1-R^2$ (log scale)")
    fig.tight_layout()
    fig.savefig(output / "transfer.png")
    plt.close(fig)
    for path in [root / name for name in ("ode_resolution/metrics.json", "ode_interpolation/metrics.json",
        "research_controls/primary_class_selection.json", "research_controls/declared_h2_and_counterchecks.json",
        "research_controls/descriptive.json", "audit_review.json")] + [extra / "frozen_transfer/metrics.json"]:
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for dataset in DATASETS:
        for path in (root / "remaining_baselines" / f"{dataset}_out_of_series/pca.json",
                     root / "ode_full_rollouts" / dataset / "trajectories.npz"):
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    facts = {"primary_class": selection, "short_results": short, "dense_results": interpolation,
             "input_sha256": sources, "test_evaluated": False}
    (paper / "facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    print("Built", len(all_rows), "rounded metric rows, seven tables and four figures")


if __name__ == "__main__":
    main()
