import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Show a readable prefix of fixed hidden-state rollout examples")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 256:
        raise ValueError("Choose a prefix between 1 and 256 analysis steps")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, dataset in zip(axes, ("Lorenz", "Rossler", "electricity")):
        directory = Path(args.run_dir) / "core" / dataset
        for name, label, color in (("observed", "Observed", "#333333"), ("ridge_0", "Ridge", "#2274a5"),
                                   ("mlp_2026", "MLP", "#e08723")):
            with np.load(directory / (name + "_paths.npz")) as data:
                values = data["paths"][0, :args.steps + 1]
            axis.plot(values[:, 0], values[:, 1], color=color, linewidth=1, alpha=.8, label=label)
        axis.scatter(values[0, 0], values[0, 1], s=20, color="black", zorder=5, label="Initial state")
        axis.set(title="Electricity" if dataset == "electricity" else dataset, xlabel="Training PC1", ylabel="Training PC2")
        axis.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .12, 1, 1))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    main()
