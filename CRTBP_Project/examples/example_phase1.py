# examples/example_phase1.py
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crtbp.constants import MU_EARTH_MOON
from crtbp.propagate import propagate, make_event_y_equals_0, make_event_too_close
from crtbp.dynamics import jacobi_constant


def main():
    mu = MU_EARTH_MOON

    # Starter nondimensional initial condition (simple moving trajectory)
    s0 = np.array([0.80, 0.0, 0.0, 0.0, 0.25, 0.0], dtype=float)

    # Our "too close" radius for Earth/Moon
    rmin = 1e-3
    
    events = [
        make_event_y_equals_0(direction=0),
        make_event_too_close(mu, rmin=rmin),
    ]

    sol = propagate(s0, (0.0, 20.0), mu, events=events, dense_output=True)

    if not sol.success:
        print("Integration failed:", sol.message)
        return

    t0, tf = sol.t[0], sol.t[-1]
    t_grid = np.linspace(t0, tf, 4000) #4000 can be increased for smoother curves

    print("dense_output:", sol.sol)
    Yg = sol.sol(t_grid).T  # shape: (N, 6)

    # Jacobi drift
    C0 = jacobi_constant(Yg[0], mu)
    C = np.array([jacobi_constant(si, mu) for si in Yg])
    drift = C - C0
    
    # Distances to primaries
    x, y, z = Yg[:, 0], Yg[:, 1], Yg[:, 2]
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)
    
    # Event counts
    n_ycross = 0
    if sol.t_events is not None and len(sol.t_events) > 0 and sol.t_events[0] is not None: n_ycross = len(sol.t_events[0])
    
    max_abs_drift = float(np.max(np.abs(drift)))
    rms_drift = float(np.sqrt(np.mean(drift**2)))

    print("Integration success:", sol.success)
    print("Max |Jacobi drift|:", float(np.max(np.abs(drift))))
    if sol.t_events is not None and len(sol.t_events) > 0:
        print("y=0 crossings:", len(sol.t_events[0]))

    print("\n--- CRTBP Phase 1 Run Summary ---")
    print("success:", sol.success)
    print("t_span:", (float(t0), float(tf)))
    print("saved solver points:", len(sol.t))
    print("uniform sample points:", len(t_grid))
    print("y=0 crossings:", n_ycross)
    print("C0:", float(C0))
    print("max |C - C0|:", max_abs_drift)
    print("RMS  |C - C0|:", rms_drift)
    print("min r1 (to Earth):", float(np.min(r1)))
    print("min r2 (to Moon): ", float(np.min(r2)))
    print("--------------------------------\n")
    
    earth_xy = (-mu, 0)
    moon_xy = (1 - mu, 0)

    # Plot trajectory in rotating frame
    plt.figure()
    plt.plot(Yg[:, 0], Yg[:, 1], label = "trajectory")
    
    # Primaries
    plt.scatter([earth_xy[0]], [earth_xy[1]], marker="o", label="Earth (m1)")
    plt.scatter([moon_xy[0]], [moon_xy[1]], marker="o", label="Moon (m2)")
    
    # Plot safety circles (rmin)
    ax = plt.gca()
    ax.add_patch(Circle(earth_xy, rmin, fill=False))
    ax.add_patch(Circle(moon_xy, rmin, fill=False))
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("CRTBP trajectory (rotating frame)")
    plt.axis("equal")
    plt.legend()

    # Plot Jacobi drift
    plt.figure()
    plt.plot(t_grid, drift)
    plt.xlabel("t")
    plt.ylabel("C - C0")
    plt.title("Jacobi drift")

    plt.show()


if __name__ == "__main__":
    main()