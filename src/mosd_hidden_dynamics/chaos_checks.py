import numpy as np
from scipy.integrate import solve_ivp


def lyapunov_spectrum(rhs, jacobian, initial, period, burn_periods=50, periods=200, intervals_per_period=8):
    initial = np.asarray(initial, dtype=np.float64)
    dimension = len(initial)
    burn_time = burn_periods * period
    if burn_time:
        solution = solve_ivp(lambda t, x: rhs(x, t), (0, burn_time), initial, method="DOP853",
                             rtol=1e-9, atol=1e-11, max_step=period / 32)
        if not solution.success:
            raise RuntimeError(solution.message)
        initial = solution.y[:, -1]
    q, sums = np.eye(dimension), np.zeros(dimension)
    dt = period / intervals_per_period
    checkpoints = []
    for step in range(periods * intervals_per_period):
        def variational(t, value):
            x = value[:dimension]
            basis = value[dimension:].reshape(dimension, dimension)
            return np.concatenate([np.asarray(rhs(x, t)), (jacobian(x, t) @ basis).reshape(-1)])
        start = burn_time + step * dt
        solution = solve_ivp(variational, (start, start + dt), np.concatenate([initial, q.reshape(-1)]),
                             method="DOP853", rtol=1e-9, atol=1e-11, max_step=period / 32)
        if not solution.success:
            raise RuntimeError(solution.message)
        initial = solution.y[:dimension, -1]
        q, r = np.linalg.qr(solution.y[dimension:, -1].reshape(dimension, dimension))
        sums += np.log(np.abs(np.diag(r)).clip(1e-300))
        if (step + 1) % (intervals_per_period * max(1, periods // 4)) == 0:
            checkpoints.append({"elapsed_periods": (step + 1) / intervals_per_period,
                                "exponents_per_time": (sums / ((step + 1) * dt)).tolist()})
    return {"exponents_per_time": (sums / (periods * period)).tolist(), "convergence": checkpoints,
            "burn_periods": burn_periods, "measurement_periods": periods,
            "qr_intervals_per_period": intervals_per_period, "method": "DOP853 variational flow with periodic QR",
            "interpretation": "Finite-time estimates; positive largest exponent supports the chosen chaotic regime."}


def finite_jacobian(rhs, value, time):
    value = np.asarray(value, dtype=np.float64)
    result = np.empty((len(value), len(value)))
    for coordinate in range(len(value)):
        epsilon = 1e-6 * max(1, abs(value[coordinate]))
        delta = np.zeros_like(value)
        delta[coordinate] = epsilon
        result[:, coordinate] = (np.asarray(rhs(value + delta, time)) - np.asarray(rhs(value - delta, time))) / (2 * epsilon)
    return result
