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
