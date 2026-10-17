# Hidden-state evolution in time series foundation models

Research code for forecasting hidden states of frozen Chronos-2 on chaotic,
seasonal and control time series.

Start with the [latest results](results/study_20261017/README.md) and
[conclusions](docs/conclusions.md). The [study protocol](docs/study_protocol.md)
defines the comparisons; [data notes](docs/data.md) explain the inputs.
The [first baseline](results/checkpoint_20261011/README.md) remains available.

## Setup

Verified on Linux with Python 3.12 and CUDA 12.8:

```bash
git clone https://github.com/nikita-aleksandrov-wr/hidden-state-evolution-analysis project/code
cd project
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r code/requirements.txt
.venv/bin/python -m pip check
cd code
```

## Smoke run

Use an allocated GPU and a fresh run ID:

```bash
../.venv/bin/python -m unittest discover -s tests -v
../.venv/bin/python scripts/run_smoke.py --config configs/smoke_chronos2_lorenz.json \
  --workspace .. --run-id example_smoke --device cuda:0
../.venv/bin/python scripts/audit_run.py --run-dir ../artifacts/example_smoke
```

Use `configs/baseline_chronos2_lorenz.json` for the original baseline.
Its [protocol](docs/experiment_protocol.md) documents state extraction and metrics.

## Extended study

The study downloads the selected electricity data and caches model states:

```bash
../.venv/bin/python scripts/run_study.py --run-id example_study --stage prepare --device cuda:0
../.venv/bin/python scripts/run_study.py --run-id example_study --stage extract --device cuda:0
../.venv/bin/python scripts/run_study.py --run-id example_study --stage evaluate --device cuda:0
```

Settings are in `configs/study_hidden_dynamics.json`. The
[study protocol](docs/study_protocol.md) includes commands for longer MLP
forecasts, frozen transfer, analog controls, 256-step rollouts, audits and figures.
Final test scores are not computed.

## Files

| Location | Contents |
| --- | --- |
| `src/`, `scripts/`, `configs/`, `tests/` | Code, declared settings and checks; included in Git |
| `docs/`, `results/` | Documentation, rounded summaries and figures; included in Git |
| `../artifacts/<run-id>/` | Data, states, precise metrics, fitted models and logs |
| `../cache/`, `../.venv/` | Downloads and local environment |

Generated arrays, weights, caches and environments stay outside Git.
Only selected summaries and figures are allowed in `results/`.
