# Experiment protocol

This checkpoint predicts one-step hidden-state evolution on noise-free Lorenz
data. Both configs use seed 2026 and the same extraction and evaluation rules.

| Config | Observations | Context length | Purpose |
| --- | ---: | ---: | --- |
| `smoke_chronos2_lorenz.json` | 8192 | 128 | Execution and repeatability |
| `baseline_chronos2_lorenz.json` | 32768 | 512 | Preliminary baseline |

## State extraction

Frozen Chronos-2 receives a separate past-only context ending at each state
endpoint. Endpoints advance by 16 raw samples, one patch stride. We capture the
last observed patch at encoder block 5, after the block and before final
LayerNorm. Block 6 is saved for a later depth comparison; indices are zero-based.

Native encoding retains REG and a masked output placeholder, but supplies no
future values. The selected state excludes these tokens. Distinct attention
group IDs prevent separate windows from exchanging information within a batch.
Native context normalization is preserved.

Inference uses float32, evaluation mode and no gradients; states are stored in
float16. Regression and metrics use float64.

## Splits

Let `B1=floor(0.7*N)` and `B2=floor(0.8*N)`. With context length `C`, eligible
source and target contexts must fit entirely within these half-open intervals:

- Train: `[0, B1)`.
- Validation: `[B1+C, B2)`.
- Test: `[B2+C, N)`.

This reserves `C` samples after each boundary; later partitions also need enough
samples to form their first complete context. The saved manifest records every
candidate pair, its membership and exclusions.

The audit checks alignment, unique indices, nonempty splits, receptive-field gaps
and exact duplicate windows across splits. Near-duplicate review remains a next
step. Test membership is audited; test states and scores are not computed.

## Prediction and metrics

Train mean predicts the mean training target; persistence predicts the current
state. Ridge uses an intercept, SVD solver and a train-fitted input StandardScaler;
targets remain unscaled.

Choose ridge alpha from `[0.0001, 0.001, 0.01, 0.1, 1, 10, 100]` by maximum
validation R², breaking ties toward smaller alpha. Shuffled-target ridge permutes
training targets and reuses the selected alpha without retuning.

```text
R² = 1 - sum((prediction - target)²) / sum((target - train_target_mean)²)
```

Cosine similarity is averaged over validation pairs. Report absolute scores and
gains over persistence. These validation scores also select alpha; final test
evaluation follows protocol freeze. Full project claims require the additional
experiments listed in the [conclusions](conclusions.md).
