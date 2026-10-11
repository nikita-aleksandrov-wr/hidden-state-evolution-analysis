# Conclusions

The extraction and baseline pipeline runs reproducibly: 12 tests, split audits
and an exact smoke repeat passed. States use past-only contexts; scaling uses
train data only, and test scores remain reserved.

Ridge outperforms persistence and train mean in both validation metrics on this
Lorenz trajectory. See the [results](../results/checkpoint_20261011/README.md).

This is preliminary evidence for linear hidden-state prediction in one
configuration. Generalization, H1/H2, pretraining effects and Markov/autonomous
dynamics require further experiments. Shuffled-target ridge has negative R²;
its null distribution still needs review.

## Claims and experiments

| Question | Evidence needed | Current status |
| --- | --- | --- |
| Does the pipeline run reproducibly? | Setup, tests, smoke run, split audit and exact repeat | Passed; 12 tests and matching smoke outputs |
| Do states use the intended observed patch? | Token, layer, past-only and batch-isolation checks | Passed for this implementation |
| Can ridge predict these Lorenz states? | Validation R²/cosine and gains over persistence | Ridge improves both; one trajectory/configuration |
| H1: does predictability generalize? | Seasonal/chaotic families, in-series/out-of-series evaluation, smoothness references, ridge/MLP gains ≥0.10 in both metrics | Pending |
| H2: is series evolution distinct from depth evolution? | Matched maps evaluated against the same future state; gains ≥0.05 in both metrics with positive paired 95% bootstrap bounds | Pending; next block only captured |
| Does pretraining explain the structure? | Matched random-initialized model and preprocessing controls | Pending |

See the [next steps](remaining_plan.md) for the remaining experiments.
