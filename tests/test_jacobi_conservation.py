import numpy as np

from crtbp.constants import MU_EARTH_MOON
from crtbp.propagate import propagate
from crtbp.dynamics import jacobi_constant


def test_jacobi_drift_small_short_run():
    mu = MU_EARTH_MOON

    # A mild IC that won't hit singularities
    s0 = np.array([0.80, 0.05, 0.0, 0.0, 0.20, 0.0], dtype=float)

    sol = propagate(s0, (0.0, 2.0), mu, dense_output=False)

    assert sol.success

    Y = sol.y.T
    C0 = jacobi_constant(Y[0], mu)
    drift = np.array([jacobi_constant(si, mu) - C0 for si in Y])

    # Loose threshold so the test is stable across machines/environments.
    # You can tighten later if you want.
    assert np.max(np.abs(drift)) < 1e-8