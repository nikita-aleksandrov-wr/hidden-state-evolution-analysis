# Conclusions

The [17 October study](../results/study_20261017/README.md) extends the first
Lorenz baseline to multiple trajectories, real electricity series and controls.

- **Short-horizon prediction transfers to unseen series.** One-step ridge R² is
  0.940 on Lorenz, 0.999 on Rossler and 0.519 on electricity.
- **Nonlinearity helps selectively.** At four steps, MLP improves Lorenz from
  0.685 to 0.804 R²; on electricity, ridge remains stronger under the fixed budget.
- **One-step accuracy does not guarantee a useful long rollout.** At 256 steps,
  ridge retains 0.655 R² on Rossler, reaches about zero on Lorenz and 0.041 on
  electricity. MLP recursive forecasts are worse at that horizon.
- **Predictability is not unique to pretrained representations.** Random-weight
  states also give high scores on chaotic inputs. These are different target
  representations, so their scores do not compare raw-series forecasting accuracy.
- **Recurrence matters.** Raw-window analogs already work well on Rossler.
  Removing windows with nearest-train cosine ≥0.999 retains about 6% of its
  unseen-series pairs; ridge still performs well on the retained subset.
- **A universal fitted operator is unsupported.** Frozen cross-dataset readouts
  transfer unevenly; a Lorenz map gives -4.072 R² on electricity.
- **The current H2 criterion needs strengthening.** It also passes on white
  noise, even with a constant-mean temporal predictor.

## Claims and evidence

| Question | Evidence collected | Status |
| --- | --- | --- |
| Is the pipeline reproducible and auditable? | 20 tests, artifact audit, independent ridge solver check, sampled FP16/FP32 comparison | Passed |
| Does prediction survive new-series validation? | Two split protocols, two chaotic systems and electricity | Positive, data-dependent |
| Is a linear autonomous law established? | Long rollouts, nonlinear/history readouts and frozen transfer | Not established |
| H1: are the approved predictability/smoothness claims supported? | Validation gains and descriptive comparisons collected | Final test and stronger specificity checks pending |
| H2: is temporal structure distinct from depth propagation? | Matched ridge maps scored against a common future target; noise/mean counterchecks | Numerical margin alone is insufficient |
| Does pretraining cause the observed predictability? | Three matched random initializations | No general causal conclusion |

These are exploratory validation results. The study uses one middle block and
four unseen validation series per main dataset; final test data remain reserved.
See the [next experiments](remaining_plan.md).
