# Next experiments

1. Complete near-duplicate and shuffled-control review. Add raw-series/AR(1)
   references, autocorrelation and jump-energy statistics; check float16 effects.
2. Add independent trajectories, more chaotic systems and strongly seasonal
   GIFT-Eval data. Fix provenance, splits and whole-series/system holdouts.
3. Add the two-layer MLP and horizons `k=1/4`. Compare ridge and MLP across both
   families and split protocols using the agreed validation rule.
4. Test H2 with matched series/depth predictors and 2000 paired series bootstrap
   samples. Add random-init, layer and preprocessing controls; use three seeds
   where fitting is stochastic.
5. Freeze the protocol before final test scoring. Consider post-hoc ODE fitting
   after these comparisons, evaluating fidelity and rollouts against discrete
   baselines.
