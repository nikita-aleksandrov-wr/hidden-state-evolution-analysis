import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot the first training context of each primary dataset")
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.reference_run)
    registry = json.loads((root / "series_registry.json").read_text())
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.8))
    for axis, name in zip(axes, ["Lorenz", "Rossler", "electricity"]):
        record = next(r for r in registry if r["dataset"] == name and r["role"] == "train")
        with np.load(root / "data" / (record["series_id"] + ".npz")) as data:
            values = data["observed"][:512]
        axis.plot(np.arange(len(values)), values, linewidth=1, color="#2274a5")
        axis.set_title({"Rossler": "Rossler", "electricity": "Electricity"}.get(name, name))
        axis.set_xlabel("Observation index")
        axis.set_ylabel("Observed value")
        axis.grid(alpha=.2)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    main()
