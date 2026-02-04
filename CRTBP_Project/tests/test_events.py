import numpy as np

from crtbp.constants import MU_EARTH_MOON
from crtbp.propagate import propagate, make_event_y_equals_0


def test_y0_event_fires():
    mu = MU_EARTH_MOON

    # Choose IC that will cross y=0 soon: start at y>0 with ydot<0
    s0 = np.array([0.80, 0.10, 0.0, 0.0, -0.25, 0.0], dtype=float)

    events = [make_event_y_equals_0(direction=0)]
    sol = propagate(s0, (0.0, 10.0), mu, events=events, dense_output=False)

    assert sol.success
    assert sol.t_events is not None
    assert len(sol.t_events) >= 1
    assert sol.t_events[0].size >= 1

    # Ensure it wasn't only at t=0 (shouldn't be, since y(0)=0.10)
    assert sol.t_events[0][0] > 0.0