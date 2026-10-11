# Hidden-state evolution in time series foundation models

The baseline pipeline generates Lorenz data, extracts rolling-context states from
frozen Chronos-2 and compares ridge with train mean, persistence and shuffled-target
ridge. Evaluation uses validation only; test data are reserved.

Start with the [results](results/checkpoint_20261011/README.md). The
[protocol](docs/experiment_protocol.md) explains the experiment,
[data notes](docs/data.md) define Lorenz and chaos, and the
[conclusions](docs/conclusions.md) and [next steps](docs/remaining_plan.md)
connect this checkpoint to the project.

## Setup

Verified on Linux with Python 3.12 and CUDA 12.8. Place this repository in
`project/code/`, then run from `project/`:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r code/requirements.txt
.venv/bin/python -m pip check
```

## Run

From `project/code/`:

```bash
../.venv/bin/python -m unittest discover -s tests -v
../.venv/bin/python scripts/run_smoke.py --config configs/smoke_chronos2_lorenz.json \
  --workspace .. --run-id example_smoke --device cuda:0
../.venv/bin/python scripts/audit_run.py --run-dir ../artifacts/example_smoke
```

For the baseline, replace the config with `configs/baseline_chronos2_lorenz.json`
and choose a new run ID. Use an allocated GPU. The first run downloads model
weights; subsequent runs use the cache.

To execute stages separately, use `--stage prepare`, then `--stage extract`, then
`--stage baseline`, keeping the same run ID and config. Source changes require a
new run ID.

## Files

| Location | Contents |
| --- | --- |
| `src/`, `scripts/`, `configs/`, `tests/` | Code, run settings and checks; included in Git |
| `docs/`, `results/` | Short documentation, rounded results and figures; included in Git |
| `../artifacts/<run-id>/` | Data, states, split manifest, precise metrics, fitted models and logs |
| `../cache/`, `../.venv/` | Model cache and local environment |

Generated artifacts, caches and environments stay outside the repository and are
also excluded by `.gitignore` if placed inside it. Only selected summaries and
figures are allowed in `results/`.
