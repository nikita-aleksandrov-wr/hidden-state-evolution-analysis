# Review experiments

Use the [documented setup](../README.md) and complete the
[initial study](study_protocol.md) first. Run commands from the code directory.
Keep generated data and models in the parent project directory.
Set CUDA_VISIBLE_DEVICES to an allocated physical GPU; scripts then use logical cuda:0.

## Comparisons

- Common-anchor baselines: state ridge/MLP, eight-state history, current observed
  patch, PCA95 affine DMD and VAR orders 1/2/4/8. MLP variants have approximately
  matched parameter counts. All use train-only transforms and macro validation selection.
- ODE: one 256-unit Tanh hidden layer, four-step segments, RK4, 60 epochs,
  patience 10 and three seeds. The foundation model stays frozen.
- Residual maps: same field capacity, Euler updates, training segments 1/4/8,
  identical four-step model selection. Ridge spectral caps retain the source intercept.
- Dense forecasts: fresh past-only states every four observations over a four-step
  interval. Ridge bridges join predicted endpoints; no observed future endpoint is supplied.
- Long forecasts: frozen fits, identical eligible anchors through 256 steps,
  fixed 50-start visual/concentration subset and train-only PCA.
- Contexts 16/128/512/1024; blocks 0/5/11 and final normalization; observation
  noise; fixed rotary-position permutation; train-only matched AR(1);
  initial-prefix per-entity normalization; strides 64/512.
  Context/layer changes alter hidden targets. Stride changes also alter horizon and sample count.
- Additional domains: Thomas, Aizawa, solar and additional electricity meters.
  Eight training and eight unseen validation entities each, four reserved.
  Frozen cross-dataset maps are scored without target adaptation.
- Solver refinement: frozen fields at 1/2/4/8/16 RK4 substeps.
  Dense and 256-step comparisons also use finer integration.

The H2 class selection follows the charter's minimum-gain rule on validation only.
The report preserves the original depth comparison and adds constant-mean and
white-noise counterexamples. These are not post-hoc replacements for H2.
Finite-time Lyapunov checks concern the physical synthetic systems.

## Commands

Use fresh run IDs. The initial-study commands create the reference run first.
The following sequence then reproduces the review extensions on one allocated GPU:

```bash
PY=../.venv/bin/python
RUN=../artifacts/example_study
ADD=../artifacts/example_additional

$PY scripts/run_remaining_baselines.py --run-dir "$RUN" --device cuda:0
$PY scripts/run_neural_ode.py --run-dir "$RUN" --device cuda:0
$PY scripts/run_residual_training.py --run-dir "$RUN" --device cuda:0
$PY scripts/run_control_mlp.py --run-dir "$RUN" --device cuda:0
$PY scripts/run_ablations.py --run-dir "$RUN" --stage extract --device cuda:0
$PY scripts/run_ablations.py --run-dir "$RUN" --stage evaluate --device cuda:0
for stage in prepare extract evaluate transfer; do
  $PY scripts/run_additional_domains.py --run-id example_additional \
    --reference-run "$RUN" --stage "$stage" --device cuda:0
done
$PY scripts/run_ode_interpolation.py --run-dir "$RUN" --device cuda:0
$PY scripts/run_ode_full_rollouts.py --run-dir "$RUN" --device cuda:0
$PY scripts/check_ode_resolution.py --run-dir "$RUN" --device cuda:0
$PY scripts/run_matched_root_search.py --run-dir "$RUN" --device cpu
$PY scripts/check_chaotic_regimes.py --main-run "$RUN" --additional-run "$ADD"
$PY scripts/analyze_research_controls.py --run-dir "$RUN"
$PY scripts/audit_review.py --run-dir "$RUN" --additional-run "$ADD"
$PY scripts/build_review_report.py --run-dir "$RUN" --additional-run "$ADD" \
  --output-dir ../artifacts/example_report --paper-data-dir ../artifacts/example_tables
```

The original study's audit and independent solver check remain required.
The review audit verifies completed cases, raw/feature hashes, common anchors,
embargoes and unopened test identities. Dense FP32 states are compared with cached
FP16 states within storage precision, rather than required to be byte-identical.

The 64-step and 256-step comparisons use different eligible subsets; methods
within each comparison share anchors. Seeds are averaged within entities, then
entities are weighted equally. Confidence intervals resample entities.
All current results are validation-only. The saved manifests and exact metrics
remain outside Git; only selected rounded summaries and figures are published.
