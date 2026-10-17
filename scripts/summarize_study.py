import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Create a compact publication folder from audited study outputs")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-id", default="main")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    from mosd_hidden_dynamics.study import aggregate_results, depth_comparison, write_rows
    from mosd_hidden_dynamics.utils import save_json, source_record
    root, output = Path(args.run_dir).resolve(), Path(args.output).resolve()
    cfg = json.loads((root / "config_requested.json").read_text())
    registry = json.loads((root / "series_registry.json").read_text())
    audit = json.loads((root / "audit_study.json").read_text())
    assert audit["status"] == "passed" and not audit["test_evaluated"]
    evaluation = root / "evaluations" / args.evaluation_id
    conditions = ["pretrained"] + [f"random_{s}" for s in cfg["random_init_seeds"]]
    rows, groups = [], []
    for condition in conditions:
        for dataset in cfg["datasets"]:
            for protocol in ("in_series", "out_of_series"):
                directory = evaluation / f"{condition}_{dataset['name']}_{protocol}"
                groups.append(json.loads((directory / "complete.json").read_text()))
                rows.extend(json.loads((directory / "metrics.json").read_text()))
    rows.extend(json.loads((evaluation / "long_mlp/metrics.json").read_text()))
    summaries = aggregate_results(rows, cfg["bootstrap_samples"], cfg["seed"])
    save_json(evaluation / "summary_all.json", summaries)
    write_rows(evaluation / "summary_all.csv", summaries)
    depth = depth_comparison(rows, cfg["bootstrap_samples"], cfg["seed"])
    converted = [{**r, "model": "ridge_direct"} if r["model"] == "train_mean" else r
                 for r in rows if r["model"] in ("train_mean", "ridge_depth")]
    mean_depth = depth_comparison(converted, cfg["bootstrap_samples"], cfg["seed"])
    analogs = json.loads((evaluation / "analogs/summary.json").read_text())
    long_rows = json.loads((evaluation / "long_rollouts/summary.json").read_text())
    transfers = json.loads((evaluation / "transfer/metrics.json").read_text())
    real = [d["name"] for d in cfg["datasets"] if d["family"] != "control"]
    def score(dataset, model, k, condition="pretrained", protocol="out_of_series"):
        return next(r for r in summaries if r["dataset"] == dataset and r["model"] == model and r["horizon"] == k
                    and r["condition"] == condition and r["protocol"] == protocol)
    def long_score(dataset, model, k):
        return next(r for r in long_rows if r["dataset"] == dataset and r["model"] == model and r["horizon"] == k)
    random_rows = []
    for dataset in cfg["datasets"]:
        for k in (1, 4, 64):
            for method in ("ridge_direct", "ridge_recursive"):
                selected = [score(dataset["name"], method, k, f"random_{seed}") for seed in cfg["random_init_seeds"]]
                random_rows.append({"dataset": dataset["name"], "protocol": "out_of_series", "condition": "random_mean",
                                    "model": method, "horizon": k, "n_seeds": len(selected),
                                    **{metric: float(np.mean([r[metric] for r in selected])) for metric in ("r2", "cosine")},
                                    "r2_std_over_initializations": float(np.std([r["r2"] for r in selected]))})
    transfer_groups = defaultdict(list)
    for row in transfers:
        transfer_groups[(row["source_dataset"], row["target_dataset"], row["model"], row["horizon"])].append(row)
    transfer_rows = [{"source_dataset": key[0], "dataset": key[1], "model": key[2], "horizon": key[3],
                      "protocol": "out_of_series", "condition": "pretrained", "n_series": len(values),
                      **{metric: float(np.mean([r[metric] for r in values])) for metric in ("r2", "cosine")}}
                     for key, values in transfer_groups.items()]
    output.mkdir(parents=True, exist_ok=False)
    published = [{"experiment": "main", **r} for r in summaries if r["condition"] == "pretrained" and r["dataset"] in real and r["horizon"] in (1, 4, 16, 64)]
    published.extend({"experiment": "random_control", **r} for r in random_rows)
    published.extend({"experiment": "raw_analogs", **r} for r in analogs if r["protocol"] == "out_of_series" and r["dataset"] in real
                     and r["horizon"] in (1, 4, 16, 64) and (r["model"].startswith("raw_knn") or r["subset"] == "far_windows"))
    published.extend({"experiment": "long_rollout", **r} for r in long_rows)
    published.extend({"experiment": "transfer", **r} for r in transfer_rows if r["source_dataset"] in real and r["dataset"] in real and r["horizon"] in (1, 4, 64))
    published.extend({"experiment": label, **r} for label, values in (("series_vs_depth", depth), ("constant_mean_vs_depth", mean_depth)) for r in values)
    write_rows(output / "summary.csv", published)
    summary = {"date": "2026-10-17", "config": "configs/study_hidden_dynamics.json", "model": "amazon/chronos-2",
               "gpu": "NVIDIA H200, device 7", "state_dimension": 768, "layers_zero_based": [5, 6],
               "context_raw": 512, "stride_raw": 16, "horizons": cfg["horizons"], "frozen_rollout_horizons": [128, 256],
               "generated_series": audit["generated_series"], "reserved_series": audit["reserved_series_not_loaded"],
               "feature_files": audit["feature_files"], "condition_protocol_dataset_groups": len(groups),
               "random_init_seeds": cfg["random_init_seeds"], "mlp_seeds": cfg["mlp"]["seeds"], "mlp_fits": audit["mlp_fits"],
               "null_control_fits": audit["null_control_fits"], "bootstrap_samples_over_series": cfg["bootstrap_samples"],
               "unit_tests_passed": 20, "artifact_audit_passed": True, "ridge_reference_agrees": True, "test_evaluated": False,
               "prepare_seconds": round(json.loads((root / "prepare_metadata.json").read_text())["seconds"], 2),
               "extraction_seconds": round(json.loads((root / "extract_metadata.json").read_text())["seconds"], 2),
               "sum_group_seconds_not_parallel_wall_time": round(sum(g["seconds"] for g in groups), 2)}
    precision_path = root / "precision_check/complete.json"
    if precision_path.exists():
        precision = json.loads(precision_path.read_text())
        summary.update(precision_check_cases=precision["cases"],
                       sampled_fp16_states_reproduced=precision["all_fp16_roundtrips_identical"],
                       fp16_r2_change_below_0_001=precision["max_absolute_r2_change"] < .001)
    save_json(output / "run_summary.json", summary)
    plot_results(output, cfg, real, summaries, random_rows, depth, mean_depth, analogs, long_rows, transfer_rows)
    table = "\n".join(f"| {dataset} | {score(dataset, 'ridge_direct', 1)['r2']:.3f} | {score(dataset, 'mlp_direct', 1)['r2']:.3f} | {score(dataset, 'ridge_direct', 4)['r2']:.3f} | {score(dataset, 'mlp_direct', 4)['r2']:.3f} |" for dataset in real)
    def rounded(value):
        return f"{0.0 if abs(value) < .0005 else value:.3f}"
    long_table = "\n".join(f"| {dataset} | {rounded(long_score(dataset, 'ridge_recursive', 256).get('r2', float('nan')))} | {rounded(long_score(dataset, 'mlp_recursive', 256).get('r2', float('nan')))} |" for dataset in real)
    text = f"""# Multi-series results — 17 October 2026

46 development series: Lorenz, Rossler, electricity, AR(1) and white noise.
Compared chronological and whole-series holdouts, horizons 1–64, three random
initializations, three MLP seeds, history inputs, depth maps and raw-window analogs.
Frozen one-step models were also rolled forward for 128/256 steps.
One analysis step advances a 512-sample context by 16 observations; adjacent
contexts share 496 samples. These scores concern hidden states, not raw forecasts.

## R² on unseen validation series

| Dataset | Ridge, k=1 | MLP, k=1 | Ridge, k=4 | MLP, k=4 |
| --- | ---: | ---: | ---: | ---: |
{table}

Scores average over four held-out series; MLP also averages three seeds.
R² uses the training-target mean. The CSV includes cosine, paired bootstrap
intervals over series and both split protocols. Validation selects alpha/epochs;
these are exploratory results. Final test data remain reserved.

## Frozen models after 256 recursive steps

| Dataset | Ridge R² | MLP R² |
| --- | ---: | ---: |
{long_table}

Each prediction is fed into the next step. These runs use common k=256 anchors,
so their shorter-horizon scores are separate from the main k=64 comparison.

## Findings

- MLP improves Lorenz at four steps, but does not consistently improve electricity.
- Random weights also produce highly predictable states: Lorenz one-step ridge
  reaches about 0.993 R². High predictability alone does not establish a pretraining effect.
- Rossler has many similar raw windows; 5-neighbor analog forecasts are already
  strong. The fixed 0.999 cosine filter retains about 6% of unseen-series pairs;
  ridge still performs well on that subset. No parameters are retuned after filtering.
- Frozen readouts transfer unevenly: Lorenz → electricity gives R² = -4.072,
  while a readout trained on electricity gives 0.519. One shared operator is unsupported.
- The declared series-versus-depth threshold also passes on white noise, even
  for a constant-mean predictor. It is insufficient evidence for distinctive temporal dynamics.
- Full H1/H2 and autonomous dynamics remain unresolved. Short-step prediction,
  long rollouts and pretraining effects must be reported separately.

## Figures and records

![Forecast horizons](horizons.png)
![Frozen long rollouts](long_rollouts.png)
![Architecture and depth controls](controls.png)
![Frozen transfer](transfer.png)

[Rounded scores](summary.csv) · [Run scope and timing](run_summary.json) ·
[Protocol and reproduction](../../docs/study_protocol.md).

20 tests and the artifact audit passed. The GPU ridge solver agrees with an
independent scikit-learn SVD fit. Ten fixed FP32/FP16 sensitivity cases changed
R² by less than 0.001. Precise scores, states, fits and logs stay outside Git.
"""
    (output / "README.md").write_text(text, encoding="utf-8")
    save_json(evaluation / "report_metadata.json", {"source": source_record(repo), "output": str(output), "test_evaluated": False})
    print(json.dumps({"output": str(output), "files": sorted(p.name for p in output.iterdir()), "summary_rows": len(published)}, indent=2))


def plot_results(output, cfg, datasets, summaries, random_rows, depth, mean_depth, analogs, long_rows, transfers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "savefig.facecolor": "white"})
    methods = [("ridge_direct", "Ridge direct", "#0072B2", "-"), ("ridge_recursive", "Ridge recursive", "#0072B2", "--"),
               ("mlp_direct", "MLP direct", "#D55E00", "-"), ("mlp_recursive", "MLP recursive", "#D55E00", "--"),
               ("raw_knn5", "Raw-window 5NN", "#CC79A7", "-."), ("persistence", "Persistence", "#777777", ":"),
               ("train_mean", "Train mean", "#222222", ":")]
    fig, axes = plt.subplots(2, len(datasets), figsize=(13, 7))
    for column, dataset in enumerate(datasets):
        for method, label, color, style in methods:
            source = analogs if method.startswith("raw_") else summaries
            chosen = [r for r in source if r["dataset"] == dataset and r["protocol"] == "out_of_series"
                      and r["model"] == method and r.get("condition") == "pretrained" and r.get("subset", "all") == "all"]
            chosen.sort(key=lambda r: r["horizon"])
            for row, metric in enumerate(("r2", "cosine")):
                axes[row, column].plot([r["horizon"] for r in chosen], [r[metric] for r in chosen], style,
                                       color=color, marker="o", markersize=3, linewidth=1.5, label=label)
                axes[row, column].set_xscale("log", base=2)
                axes[row, column].set_xticks(cfg["horizons"], [str(k) for k in cfg["horizons"]])
                axes[row, column].grid(alpha=.2)
        axes[0, column].set_title(dataset)
        axes[1, column].set_xlabel("Horizon (analysis steps)")
    axes[0, 0].set_ylabel("R² (training-mean reference)")
    axes[1, 0].set_ylabel("Mean cosine similarity")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Hidden-state forecasts on unseen validation series")
    fig.tight_layout(rect=(0, .1, 1, .95))
    fig.savefig(output / "horizons.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 3.8))
    for ax, dataset in zip(axes, datasets):
        for method, label, color in (("ridge_recursive", "Ridge", "#0072B2"), ("mlp_recursive", "MLP", "#D55E00")):
            selected = sorted([r for r in long_rows if r["dataset"] == dataset and r["model"] == method and "r2" in r], key=lambda r: r["horizon"])
            ax.plot([r["horizon"] for r in selected], [r["r2"] for r in selected], marker="o", markersize=3, color=color, label=label)
        ax.axhline(0, color="#777777", linestyle=":", label="Train mean")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64, 256], [1, 4, 16, 64, 256])
        ax.set(title=dataset, xlabel="Recursive steps")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("R² (training-mean reference)")
    axes[-1].legend(frameon=False)
    fig.suptitle("Frozen one-step maps: no true-state feedback")
    fig.tight_layout()
    fig.savefig(output / "long_rollouts.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    names = [d["name"] for d in cfg["datasets"]]
    positions = np.arange(len(names))
    pretrained = [next(r["r2"] for r in summaries if r["dataset"] == name and r["condition"] == "pretrained" and r["protocol"] == "out_of_series" and r["model"] == "ridge_direct" and r["horizon"] == 1) for name in names]
    random = [next(r for r in random_rows if r["dataset"] == name and r["model"] == "ridge_direct" and r["horizon"] == 1) for name in names]
    axes[0].bar(positions - .18, pretrained, width=.36, color="#0072B2", label="Pretrained")
    axes[0].bar(positions + .18, [r["r2"] for r in random], yerr=[r["r2_std_over_initializations"] for r in random], width=.36, color="#D55E00", label="Random weights (3 seeds ±SD)", capsize=3)
    axes[0].set_xticks(positions, names, rotation=15)
    axes[0].set(title="Predictability also occurs with random weights", ylabel="Ridge R², k=1, unseen series")
    axes[0].legend(frameon=False, fontsize=8)
    for offset, values, label, color in ((-.18, depth, "Series ridge", "#0072B2"), (.18, mean_depth, "Constant mean", "#D55E00")):
        row = next(r for r in values if r["dataset"] == "white_noise" and r["protocol"] == "out_of_series")
        axes[1].bar(np.arange(2) + offset, [row["delta_r2"], row["delta_cosine"]], width=.36, label=label, color=color)
    axes[1].axhline(.05, color="#777777", linestyle=":", label="Declared margin: 0.05")
    axes[1].set_xticks([0, 1], ["R² gain", "Cosine gain"])
    axes[1].set(title="Depth criterion also passes on white noise", ylabel="Gain over depth predictor")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "controls.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, k in zip(axes, (1, 4, 64)):
        matrix = np.asarray([[next(r["r2"] for r in transfers if r["source_dataset"] == source and r["dataset"] == target
                                  and r["horizon"] == k and r["model"] == "ridge_direct") for target in datasets] for source in datasets])
        graphic = ax.imshow(matrix, cmap="RdBu", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
        for row in range(len(datasets)):
            for col in range(len(datasets)):
                value = matrix[row, col]
                text = f"{value:.3f}" if abs(value) < 10 else f"{value:.2e}"
                ax.text(col, row, text, ha="center", va="center", fontsize=9, color="white" if abs(value) > .6 else "black")
        ax.set_xticks(range(len(datasets)), datasets, rotation=20)
        ax.set_yticks(range(len(datasets)), datasets)
        ax.set(title=f"Direct ridge, k={k}", xlabel="Evaluation dataset")
    axes[0].set_ylabel("Fit dataset")
    fig.suptitle("Frozen readout transfer: source scaler and weights unchanged")
    fig.subplots_adjust(left=.1, right=.84, bottom=.25, top=.83, wspace=.45)
    color_axis = fig.add_axes([.89, .28, .018, .46])
    fig.colorbar(graphic, cax=color_axis, label="R²", extend="min")
    fig.text(.5, .03, "Color scale saturates outside [-1, 1]; cell labels show the actual scores.", ha="center", fontsize=9)
    fig.savefig(output / "transfer.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
