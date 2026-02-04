# propagate.py
import numpy as np
from scipy.integrate import solve_ivp
from .dynamics import crtbp_eom

def make_event_y_equals_0(direction=0):
    def event(t, s):
        return s[1]  # y
    event.terminal = False
    event.direction = direction
    return event

def make_event_too_close(mu, rmin=1e-3):
    # Stops if we get too close to either primary
    def event(t, s):
        x, y, z = s[0], s[1], s[2]
        r1 = np.linalg.norm([x + mu, y, z])
        r2 = np.linalg.norm([x - (1 - mu), y, z])
        return min(r1, r2) - rmin
    event.terminal = True
    event.direction = -1
    return event

def propagate(
    state0,
    t_span,
    mu: float,
    rtol: float = 1e-11,
    atol: float = 1e-13,
    events=None,
    dense_output: bool = True,
    max_step: float = np.inf,
):
    state0 = np.asarray(state0, dtype=float)

    def fun(t, s):
        return crtbp_eom(t, s, mu)

    sol = solve_ivp(
        fun=fun,
        t_span=t_span,
        y0=state0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        events=events,
        dense_output=dense_output,   # <-- THIS must be here
        max_step=max_step,
    )
    return sol