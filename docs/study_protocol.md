# Multi-series study protocol

The study keeps Chronos-2 frozen and uses the same past-only state definition as
the [baseline](experiment_protocol.md): block 5, block 6 for depth controls,
context 512, stride 16, float32 inference and float16 state storage.
One analysis step advances by 16 raw observations: adjacent contexts share 496
of 512 samples. The targets are hidden states, not future raw observations.

## Data and splits

Lorenz, Rossler and electricity each use 12 development series; eight are training
series and four are unseen validation series. Four additional series per dataset
are reserved. AR(1) and white noise each use three training, two validation and
one reserved series. Each prepared series has 32768 observations.

For **in-series** evaluation, every development series uses chronological
70/10/20 boundaries and a full-context embargo. For **out-of-series** evaluation,
training and validation identities are disjoint. Both protocols leave the last
20% of every series unused. Reserved identities are never prepared or encoded.

All methods share anchors eligible for the largest horizon, including the past
context needed by the history model. Test states and scores are never computed.

## Models and selection

Direct ridge fits a separate map for k=1/4/8/16/32/64. Recursive ridge repeatedly
applies its k=1 map to its own predictions. History ridge uses `(h_previous, h)`.
Inputs are scaled on train; alpha is selected by mean validation R² over series.
The float64 GPU solver is checked against scikit-learn SVD.

MLP has two linear layers with GELU and width 256. It uses train-fitted input and
target scaling; its weighted training loss equals raw-target MSE up to a constant.
The budget is 100 epochs with patience 15; the checkpoint maximizes mean
validation R². All three seeds are retained. This is a fixed-budget comparison.

Each series gets vector R² relative to the pooled training-target mean and mean
cosine. MLP seeds are averaged within each series, then series receive equal
weight. Paired bootstrap intervals use 2000 resamples of whole series. They
remain validation intervals.

## Controls

- Three random initializations retain the architecture and preprocessing.
- Twenty target permutations and twenty within-series circular shifts per
  pretrained dataset/protocol reuse the selected alpha. These are sanity
  distributions, not permutation-test p-values.
- Raw-window 1NN/5NN use only training neighbors, with centered unit-length past
  contexts. The fixed cosine threshold 0.999 defines a sensitivity subset;
  selected models are evaluated there without retuning.
- The depth map fits `h_block5 -> h_block6` and selects alpha on its depth target.
  Both depth and temporal predictions are scored against the same future block-5
  state. Constant-mean and white-noise counterchecks expose limited specificity.
- Frozen cross-dataset transfer preserves source weights/scaling. The target
  training mean defines only the R² reference; it does not adapt the model.
- The 128/256-step stress test freezes the original k=1 fits and uses common
  k=256 anchors. Its shorter-horizon scores have a different evaluation subset.
- Precision checks compare fresh FP32 states with FP16 storage on ten fixed
  64-pair samples; random ridge fits are reconstructed without retuning.

## Reproduction

Run preparation, extraction and evaluation as in the [README](../README.md).
From `code/`, complete the supplementary analyses with an allocated GPU:

```bash
../.venv/bin/python scripts/run_long_mlp.py --run-dir ../artifacts/example_study --device cuda:0
../.venv/bin/python scripts/run_transfer.py --run-dir ../artifacts/example_study --device cuda:0
../.venv/bin/python scripts/run_analogs.py --run-dir ../artifacts/example_study --device cuda:0
../.venv/bin/python scripts/run_long_rollouts.py --run-dir ../artifacts/example_study --device cuda:0
../.venv/bin/python scripts/check_state_precision.py --run-dir ../artifacts/example_study --device cuda:0
../.venv/bin/python scripts/audit_study.py --run-dir ../artifacts/example_study --verify-ridge --device cuda:0
../.venv/bin/python scripts/summarize_study.py --run-dir ../artifacts/example_study --output ../artifacts/example_study/publication
```

Full arrays, checkpoints, manifests and precise records stay in `artifacts/`.
Only reviewed tables, figures and the short report belong in `results/`.
