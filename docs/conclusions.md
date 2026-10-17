# Conclusions

These are exploratory validation results from frozen Chronos-2.
Final test partitions remain closed. See the [review tables and figures](../results/review_20261017/README.md).

- **Short forecasts work on unseen series.** One-step ridge R² is 0.940 on
  Lorenz, 0.999 on Rossler and 0.519 on electricity. The extension also covers
  Thomas, Aizawa, solar and additional electricity meters.
- **Nonlinearity helps selectively.** At four steps, Lorenz improves from
  ridge 0.685 to MLP 0.806. Ridge remains stronger on electricity under the
  fixed budget. The new comparison uses common anchors with eight-state history;
  the earlier MLP value 0.804 used the initial study's anchors.
- **Smoothness is not universal.** Hidden jumps depend on the dataset and
  coordinate scaling. Electricity hidden states are less smooth than observed patches.
- **Prediction does not imply a sufficient Markov state.** History helps some
  linear fits. Adding the observed patch has a small effect in the tested setup.
- **One-step accuracy does not guarantee a useful rollout.** At 256 steps,
  ridge retains 0.655 R² on Rossler, about zero on Lorenz and 0.041 on electricity.
  The tested ODE and residual maps deteriorate substantially.
- **ODE conclusions require a solver check.** The field was trained with coarse
  RK4 steps. Its forecasts change with integration resolution; this limits any
  interpretation as a continuous evolution law.
- **The depth comparison lacks specificity.** Its threshold also passes on white
  noise and with a constant predictor. It cannot alone establish temporal dynamics.
- **Pretraining is not the only explanation.** Random-weight states can also
  be predictable. Shared windows, recurrence, layer and context choice matter.
- **A universal fitted operator is unsupported.** Frozen maps transfer unevenly.
  Scores for different hidden targets do not compare raw-series forecasting accuracy.

Remaining work: protocol review, reserved-test evaluation and peer-review revision.
No claim that hidden states are linear in time is supported.
