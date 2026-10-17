# Lorenz data

Chaotic means deterministic nonlinear dynamics with sensitive dependence on
initial conditions. The Lorenz observations follow three equations:

```text
dx/dt = sigma * (y - x)
dy/dt = x * (rho - z) - y
dz/dt = x * y - beta * z
```

We use the [dysts](https://github.com/GilpinLab/dysts) implementation, with its
parameters `sigma=10`, `rho=28`, `beta=2.667` and initial state `[1, 1, 1]`.
The system collection is described in [Gilpin, NeurIPS 2021](https://arxiv.org/abs/2110.05266).

Data preparation generates a trajectory, discards 1024 burn-in samples and keeps
8192 samples for smoke runs or 32768 for the baseline. Only coordinate `x` enters
the model. There is no added noise or trajectory standardization; `y` and `z`
are saved for inspection.

Integration uses DOP853 with `rtol=1e-9`, `atol=1e-12` and dysts Fourier resampling
at 128 points per dominant period. One model step spans 16 output samples.
Physical timestamps and generation settings are saved with the raw data.

The [trajectory figure](../results/checkpoint_20261011/lorenz_data.png) illustrates
the data. This checkpoint uses the established Lorenz regime; it does not estimate
a Lyapunov exponent. Run settings are in `configs/`; see the
[setup and stage commands](../README.md) to regenerate the data.

## Extended study

Rossler uses the same generation procedure, with dysts parameters
`a=0.2`, `b=0.2`, `c=5.7`. New trajectories use seeded perturbations of the
declared initial conditions; only coordinate `x` enters Chronos-2.

Electricity comes from the pinned `electricity/15T` configuration of
[GIFT-Eval](https://huggingface.co/datasets/Salesforce/GiftEval). The first twelve
entities are development data: MT_001–MT_008 train and MT_009–MT_012 validation
for the whole-series protocol. Leading missing values are trimmed; internal gaps
use past-only forward filling. The first 32768 samples after trimming are kept.
Pretraining overlap is unknown, so these are held-out readout entities, not a
claim of foundation-model zero-shot evaluation.

AR(1) uses coefficient 0.9 and Gaussian innovations; white noise uses coefficient
0. Both are controls, with 1024 burn-in samples and recorded seeds.
See the [study protocol](study_protocol.md) for memberships and evaluation.
