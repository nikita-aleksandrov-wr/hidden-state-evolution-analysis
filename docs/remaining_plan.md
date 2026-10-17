# Next experiments

1. Strengthen H2 so that a constant mean on white noise cannot satisfy the claimed
   evidence. Agree on the interpretation before final test scoring.
2. Compare nonlinear predictors with and without history at matched capacity.
   Add a past-observation input control to separate state sufficiency from readout limitations.
3. Test context length, layer choice and sampling scale. Investigate why random
   states are easy to predict and how much shared input history explains the scores.
4. Expand whole-series/system holdouts beyond four validation series per dataset.
   Add more varied chaotic systems and seasonal domains.
5. Compare stable/reduced-rank maps and multi-step training objectives for long
   rollouts. Consider Neural ODEs after they have a clear advantage to test.
6. Freeze the protocol, then evaluate the reserved test partitions once.

Completed experiments and their limitations are in the
[results](../results/study_20261017/README.md).
