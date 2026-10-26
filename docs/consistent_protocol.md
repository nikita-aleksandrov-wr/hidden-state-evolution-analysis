# Recursive comparison

The question is whether one fixed transition predicts future rolling-context states
of frozen Chronos-2. This comparison uses Lorenz, Rossler and electricity, with
eight training and four validation entities per dataset. Final test data are excluded.

All models use middle-block states (zero-based block 5, sixth of 12), context 512 and stride 16. Training and selection
share anchors eligible for a 64-step horizon and eight-state history support.
The models themselves receive only the current state. A separate report manifest
supports 256 steps; every reported horizon uses this same set of starts.

Ridge fits one-step pairs and selects regularization by recursive four-step
validation R². MLP, residual and ODE share a 768–256–768 Tanh architecture,
four-step raw-coordinate training loss and three seeds. Each model has its own
weights. Neural settings and budgets are in
[`consistent_hidden_dynamics.json`](../configs/consistent_hidden_dynamics.json).
ODE uses four RK4 substeps during training and its primary rollout.

## Run

Start from a prepared and extracted study produced by `scripts/run_study.py`.
The reference directory must contain its data registry, backbone configuration
and pretrained/random feature caches. Run from the code directory with an
allocated GPU. Choose a fresh output directory.

```bash
../.venv/bin/python scripts/run_consistent_grid.py \
  --reference-run ../artifacts/example_study \
  --output-dir ../artifacts/comparison/core --dataset Lorenz \
  --device cuda:0 --stop-file ../artifacts/comparison/STOP
```

Repeat for `Rossler` and `electricity`. Completed neural fits are reusable;
an interrupted unfinished fit restarts from its seed.

```bash
../.venv/bin/python scripts/run_consistent_control.py \
  --output-dir ../artifacts/comparison --device cuda:0 \
  --stop-file ../artifacts/comparison/STOP
../.venv/bin/python scripts/run_reencoding_control.py \
  --reference-run ../artifacts/example_study \
  --output-dir ../artifacts/comparison --device cuda:0 \
  --stop-file ../artifacts/comparison/STOP
../.venv/bin/python scripts/run_reencoding_null.py \
  --reference-run ../artifacts/example_study \
  --output-dir ../artifacts/comparison --device cuda:0
../.venv/bin/python scripts/build_consistent_report.py \
  --run-dir ../artifacts/comparison --output-dir ../artifacts/comparison/report
```

Input re-encoding uses the same 32 starts per validation entity for every method,
including readouts rescored on that subset. It evaluates horizons 1, 4, 16 and 64.
The pretrained median forecast and last-value continuation are encoded in pretrained
and three random-weight spaces. Each space has its own targets and train-fitted
metrics. White noise supplies a specificity check, not evidence of raw-noise predictability.

For frozen-field integration sensitivity on the same 16 path starts per entity:

```bash
../.venv/bin/python scripts/run_consistent_refinement.py \
  --reference-run ../artifacts/example_study --run-dir ../artifacts/comparison \
  --dataset Lorenz --seeds 2026 2027 2028 --device cuda:0 \
  --stop-file ../artifacts/comparison/STOP
```

Repeat for the other datasets. This compares 4, 8 and 16 substeps without refitting.
Its smaller path subset must not be mixed with the full report manifest.

## Interpret

R² uses the training-target mean, separately for each horizon. Report macro scores
across entities, seed standard deviations and entity-bootstrap intervals after
averaging seeds within entities. Four entities are a small evaluation group;
overlapping windows and repeated seeds do not increase the number of entities.
Selection and reporting reuse validation entities, so these are development results.

Pointwise accuracy, finite-path statistics and coarse boundedness answer different
questions. Passing an implementation control or beating input re-encoding does
not establish a Markov state, an invariant measure or an autonomous law.
The declared H1/H2 are unchanged. Freeze a separate procedure before final test evaluation.

The [results](../results/consistent_20261026/README.md) contain rounded tables and
figures. Data, checkpoints, precise records, environment details and source hashes
stay in the artifact directory outside Git.
