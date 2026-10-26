# Conclusions

The [26 October comparison](../results/consistent_20261026/README.md) is the current
matched grid: four transition models, three datasets, five recursive horizons,
and three seeds for each neural model. Chronos-2 remains frozen.

- **Nonlinearity helps at selected horizons.** On Lorenz at four recursive steps,
  R² is 0.658 for ridge and 0.840 ± 0.006 for MLP. The MLP advantage remains with
  coordinate-standardized and whitened-PCA scores.
- **Long prediction depends on the data and metric.** At 256 steps ridge R² is
  0.655 on Rossler, approximately zero on Lorenz and 0.055 on electricity.
  Rossler ridge falls to 0.264 in whitened PCA space. Strong recurrence and
  fixed synthetic-system parameters limit the generalization claim.
- **Point accuracy and trajectory statistics can disagree.** On Lorenz and
  Rossler, MLP better matches the measured projected distribution, ACF and PSD
  than ridge, despite worse 256-step pointwise R². These are finite-path
  diagnostics, not proof of a recovered attractor.
- **The input-forecast control is weaker in this comparison.** At four steps,
  Chronos forecasting followed by re-encoding scores -0.067, 0.791 and 0.133
  on Lorenz, Rossler and electricity. Ridge also beats this control on white
  noise, so a gain alone cannot establish meaningful autonomous dynamics.
- **Improved ODE integration does not resolve the long-rollout failure.** RK4
  with four training substeps passes a known-flow implementation control, but
  the fitted hidden-state fields still have strongly negative long-horizon R².
  This concerns the tested field, loss, solver and budget.

All new scores use validation data. The original H1/H2 and final test remain unchanged.
The ± values above describe seed standard deviations; series-level intervals
and individual-entity results are in the accompanying tables.

## Earlier experiments and controls

These are exploratory validation results from frozen Chronos-2.
Final test partitions remain closed. See the [review tables and figures](../results/review_20261017/README.md).

- **Short forecasts work on unseen series.** One-step ridge R² is 0.940 on
  Lorenz, 0.999 on Rossler and 0.519 on electricity. The extension also covers
  Thomas, Aizawa, solar and additional electricity meters.
- **Nonlinearity helps selectively.** At four steps, Lorenz improves from
  ridge 0.685 to MLP 0.806. Ridge remains stronger on electricity under the
  fixed budget. The new comparison uses common anchors supporting eight-state VAR;
  the earlier MLP value 0.804 used the initial study's anchors. The history
  readouts use the current and previous states; VAR(8) uses eight states.
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
