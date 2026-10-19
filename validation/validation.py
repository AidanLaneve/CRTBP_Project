import numpy as np
import matplotlib.pyplot as plt

from crtbp.constants import MU_EARTH_MOON

from crtbp.equilibria import lagrange_points
from crtbp.equilibria import _f_collinear

from crtbp.dynamics import crtbp_eom
from crtbp.dynamics import jacobi_constant

from crtbp.linear_stability_analysis import (
    jacobian_J,
    omega,
    distances
    )
from crtbp.equilibria import (
    lagrange_points as lagrange_points_bisection)
from crtbp.linear_stability_analysis import (
    lagrange_points as lagrange_points_newton)

from scipy.integrate import solve_ivp


mu = MU_EARTH_MOON

# Exact L4 state in the rotating CR3BP frame
state_l4 = np.array([
    0.5 - mu,              # x
    np.sqrt(3.0) / 2.0,   # y
    0.0,                   # z
    0.0,                   # vx
    0.0,                   # vy
    0.0                    # vz
    ])

rhs_l4 = crtbp_eom(
    0.0,
    state_l4,
    mu
    )

print("L4 state:")
print(state_l4)

print("\nDerivative at L4:")
print(rhs_l4)

print("\nL4 equilibrium residual:")
print(np.linalg.norm(rhs_l4))

# -------------------------------------------------
# Validation 2: Jacobi Constant at L4

x, y, z, vx, vy, vz = state_l4

r1 = np.sqrt(
    (x + mu)**2
    + y**2
    +z**2
    )

r2 = np.sqrt(
    (x - 1.0 + mu)**2
    + y**2
    + z**2
    )

C_direct = (
    x**2
    + y**2
    + 2.0 * (1.0 - mu) / r1
    + 2.0 * mu / r2
    - (vx**2 + vy**2 + vz**2)
    )

C_exact = 3.0 - mu + mu**2

error = abs(C_direct - C_exact)

print("\nr1 at L4:", r1)
print("r2 at L4:", r2)

print("\nJacobi calculated directly:", C_direct)
print("Jacobi absolute error:      ", error)

C_code = jacobi_constant(state_l4, mu)

print("\nJacobi from project function:", C_code)
print("Exact Jacobi value:         ", C_exact)
print("Function absolute error:    ", abs(C_code - C_exact))

# ----------------------------------------------------
# Validation 3: Propagation, exact L4 Equilibrium

t0 = 0.0
tf = 2.0 * np.pi

sol = solve_ivp(
    fun=lambda t, state: crtbp_eom(t, state, mu),
    t_span = (t0, tf),
    y0=state_l4,
    method = "DOP853",
    rtol = 1e-12,
    atol = 1e-12,
    max_step = 0.05
    )

initial_state = sol.y[:, 0]
final_state= sol.y[:, -1]

final_error = final_state - initial_state
final_error_norm = np.linalg.norm(final_error)

print("\n--- L4 Propagation Validation ---")

print("\nInitial state:")
print(initial_state)

print("\nFinal state:")
print(final_state)

print("\nFinal state error:")
print(final_error)

print("\nFinal state error norm:")
print(final_error_norm)

print("\nNumber of integration steps:")
print(len(sol.t))

deviation = sol.y.T - state_l4

deviation_norm = np.linalg.norm(
    deviation,
    axis = 1
    )

max_deviation = np.max(deviation_norm)

print("\nMaximum state deviation:")
print(max_deviation)

C_values = np.array([
    jacobi_constant(state, mu)
    for state in sol.y.T
    ])

C0 = C_values[0]

C_drift = C_values - C0

max_C_drift = np.max(
    np.abs(C_drift)
    )

print("\nInitial Jacobi constant:")
print(C0)

print("\nMaximum Jacobi drift:")
print(max_C_drift)

# ----------------------------------------------------
# Validation 4: Moving Trajectory Near L4

perturbed_l4 = np.array([
    0.5 - mu + 1.0e-3,
    np.sqrt(3.0) / 2.0,
    0.0,
    0.0,
    0.0,
    0.0
    ])

t0 = 0.0
tf = 20.0

sol_perturbed = solve_ivp(
    fun=lambda t, state: crtbp_eom(t, state, mu),
    t_span=(t0, tf),
    y0=perturbed_l4,
    method="DOP853",
    rtol=1e-11,
    atol=1e-12,
    max_step=0.05
    )

print("\n--- Perturbed L4 Validation ---")

print("\nInitial State:")
print(sol_perturbed.y[:, 0])

print("\nFinal State:")
print(sol_perturbed.y[:, -1])

print("\nNumber of Integration Steps:")
print(len(sol_perturbed.t))

C_values = np.array([
    jacobi_constant(state, mu)
    for state in sol_perturbed.y.T
    ])

C0 = C_values[0]

C_drift = C_values -C0

max_C_drift = np.max(np.abs(C_drift))

relative_C_drift = max_C_drift / abs(C0)

print("\nInitial Jacobi constant:")
print(C0)

print("\nMinimum Jacobi constant:")
print(np.min(C_values))

print("\nMaximum Jacobi constant:")
print(np.max(C_values))

print("\nMaximum absolute Jacobi drift:")
print(max_C_drift)

print("\nMaximum relative Jacobi drift:")
print(relative_C_drift)

plt.figure()

plt.plot(
    sol_perturbed.y[0],
    sol_perturbed.y[1]
)

plt.scatter(
    0.5 - mu,
    np.sqrt(3.0) / 2.0,
    marker="x",
    label="L4"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Perturbed Motion Near L4")
plt.axis("equal")
plt.grid()
plt.legend()

plt.show()

# ----------------------------------------------------
# Validation 5: Convergence

t_eval = np.linspace(0.0, 20.0, 2001)

reference = solve_ivp(
    fun=lambda t, state: crtbp_eom(t, state, mu),
    t_span=(0.0, 20.0),
    y0=perturbed_l4,
    method="DOP853",
    rtol=1e-13,
    atol=1e-14,
    t_eval=t_eval
)

print("\nReference function evaluations:")
print(reference.nfev)

tolerances = [
    (1e-6,  1e-8),
    (1e-8,  1e-10),
    (1e-10, 1e-12),
    (1e-12, 1e-13)
]

print("\n--- Convergence Study ---")

for rtol, atol in tolerances:

    sol_test = solve_ivp(
        fun=lambda t, state: crtbp_eom(t, state, mu),
        t_span=(0.0, 20.0),
        y0=perturbed_l4,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        t_eval=t_eval
    )

    # ---------------------------------
    # Compare against reference solution
    # ---------------------------------

    state_difference = sol_test.y - reference.y

    error_vs_reference = np.linalg.norm(
        state_difference,
        axis=0
    )

    max_state_error = np.max(error_vs_reference)

    final_state_error = np.linalg.norm(
        sol_test.y[:, -1]
        - reference.y[:, -1]
    )

    # ---------------------------------
    # Jacobi conservation
    # ---------------------------------

    C_values = np.array([
        jacobi_constant(state, mu)
        for state in sol_test.y.T
    ])

    max_C_drift = np.max(
        np.abs(C_values - C_values[0])
    )

    print(f"\nrtol = {rtol:.0e}, atol = {atol:.0e}")

    print("Function evaluations:")
    print(sol_test.nfev)

    print("Final state error:")
    print(final_state_error)

    print("Maximum state error:")
    print(max_state_error)

    print("Maximum Jacobi drift:")
    print(max_C_drift)
    
# ----------------------------------------------------
# Validation 6: Crossing Events
    
event_test_state = np.array([
    0.8,
    0.05,
    0.0,
    0.0,
    -0.15,
    0.0
    ])

def y_crossing(t, state):      #event function
    return state[1]
    
y_crossing.terminal = False
y_crossing.direction = 0

sol_event = solve_ivp(
    fun=lambda t, state: crtbp_eom(t, state, mu),
    t_span=(0.0, 10.0),
    y0=event_test_state,
    method="DOP853",
    rtol=1e-11,
    atol=1e-12,
    events=y_crossing,
    dense_output=True
    )

print("\n--- Event Detection Validation ---")

print("Number pf y=0 crossings:")
print(len(sol_event.t_events[0]))

print("\nCrossing times:")
print(sol_event.t_events[0])

print("\nCrossing states:")
print(sol_event.y_events[0])

event_states = sol_event.y_events[0]

y_residuals = np.abs(
    event_states[:, 1]
    )

print("\ny residuals at crossing:")
print(y_residuals)

print("\nMaximum event residual:")
print(np.max(y_residuals))

for i, state in enumerate(event_states):
    
    vy = state[4]
    
    if vy > 0:
        direction = "upwards"
        
    elif vy < 0:
        direction = "downwards"
        
    else:
        direction = "tangent"
        
    print(
        f"Event {i + 1}: "
        f"vy = {vy:.12e}, "
        f"crossing = {direction}"
        )

# Upward crossings
y_crossing.direction = 1

sol_up = solve_ivp(
    fun=lambda t, state: crtbp_eom(t, state, mu),
    t_span=(0.0, 10.0),
    y0=event_test_state,
    method="DOP853",
    rtol=1e-11,
    atol=1e-12,
    events=y_crossing
    )

up_states = sol_up.y_events[0]

print("\n--- Upward crossings ---")
print("Number:", len(up_states))
print("vy values:", up_states[:, 4])

assert np.all(up_states[:, 4] > 0)

# Downward crossings
y_crossing.direction = -1

sol_down = solve_ivp(
    fun=lambda t, state: crtbp_eom(t, state, mu),
    t_span=(0.0, 10.0),
    y0=event_test_state,
    method="DOP853",
    rtol=1e-11,
    atol=1e-12,
    events=y_crossing
    )

down_states = sol_down.y_events[0]

print("\n--- Downward crossings ---")
print("Number:", len(down_states))
print("vy values:", down_states[:, 4])

assert np.all(down_states[:, 4] < 0)

# ----------------------------------------------------
# Validation 7: Lagrange Points

print("\n--- Lagrange Point Equilibrium Validation ---")
points = lagrange_points(MU_EARTH_MOON)

point_names = ["L1", "L2", "L3", "L4", "L5"]

for name in point_names:
    
    position = getattr(points, name)
    
    state = np.array([
        position[0],
        position[1],
        position[2],
        0.0,
        0.0,
        0.0
        ])
    
    rhs = crtbp_eom(
        0.0,
        state,
        mu
        )
    
    residual = np.linalg.norm(rhs)
    
    print(f"\n{name}")
    print("Position:")
    print(position)
    
    print("Derivative:")
    print(rhs)
    
    print("Equilibrium residual:")
    print(residual)
    
    assert residual < 1e-10

print("\n--- Collinear Root Residuals ---")

for name in ["L1", "L2", "L3"]:
    
    position = getattr(points, name)
    
    x = position[0]
    
    root_residual = _f_collinear(
        x,
        mu
        )
    
    print(
        f"{name}: "
        f"f(x) = {root_residual:16e}"
        )
    
x_Earth = -mu
x_Moon = 1.0 - mu

assert points.L3[0] < x_Earth
assert x_Earth < points.L1[0] < x_Moon
assert points.L2[0] > x_Moon

print("\nPhysical ordering check: PASS")

assert np.isclose(points.L4[0], points.L5[0])
assert np.isclose(points.L4[1], -points.L5[1])
assert np.isclose(points.L4[2], 0.0)
assert np.isclose(points.L5[2], 0.0)

print("L4/L5 symmetry check: PASS")

# ----------------------------------------------------
# Validation 8: Analytical vs. Finite-Distance Jacobian

def numerical_jacobian(state, mu, h=1e-6):
    
    state = np.asarray(state, dtype=float)    
    n = len(state)
    
    J_fd = np.zeros((n, n))
    
    for j in range(n):
        
        perturbation = np.zeros(n)
        perturbation[j] = h
        
        f_plus = crtbp_eom(
            0.0,
            state + perturbation,
            mu
            )
        f_minus = crtbp_eom(
            0.0,
            state - perturbation,
            mu
            )
        
        J_fd[:, j] = (
            f_plus - f_minus
            ) / (2.0 * h)
        
    return J_fd

points = lagrange_points(mu)

x, y, z = points.L4

state_L4 = np.array([
    x,
    y,
    z,
    0.0,
    0.0,
    0.0
])

J_analytic = jacobian_J(
    x,
    y,
    z,
    mu
)

J_fd = numerical_jacobian(
    state_L4,
    mu
)

difference = J_analytic - J_fd

max_error = np.max(
    np.abs(difference)
)

frobenius_error = np.linalg.norm(
    difference
)

print("\n--- Jacobian Validation at L4 ---")

print("\nAnalytical Jacobian:")
print(J_analytic)

print("\nFinite-Difference Jacobian:")
print(J_fd)

print("\nDifference:")
print(difference)

print("\nMaximum absolute element error:")
print(max_error)

print("\nFrobenius norm of error:")
print(frobenius_error)

print("\n--- Finite-Difference Step Study ---")

for h in [
    1e-3,
    1e-4,
    1e-5,
    1e-6,
    1e-7,
    1e-8
]:

    J_fd = numerical_jacobian(
        state_L4,
        mu,
        h=h
    )

    difference = (
        J_analytic
        - J_fd
    )

    max_error = np.max(
        np.abs(difference)
    )

    print(
        f"h = {h:.0e}   "
        f"max error = {max_error:.6e}"
    )
    
print("\n--- Jacobian Validation at All Lagrange Points ---")

for name in [
    "L1",
    "L2",
    "L3",
    "L4",
    "L5"
]:

    position = getattr(
        points,
        name
    )

    x, y, z = position

    state = np.array([
        x,
        y,
        z,
        0.0,
        0.0,
        0.0
    ])

    J_analytic = jacobian_J(
        x,
        y,
        z,
        mu
    )

    J_fd = numerical_jacobian(
        state,
        mu,
        h=1e-6
    )

    difference = (
        J_analytic
        - J_fd
    )

    max_error = np.max(
        np.abs(difference)
    )

    frobenius_error = np.linalg.norm(
        difference
    )

    print(f"\n{name}")

    print(
        "Maximum element error:",
        max_error
    )

    print(
        "Frobenius error:",
        frobenius_error
    )

# ----------------------------------------------------
# Validation 9: Eigenvalue Symmetry

def nearest_eigenvalue_distance(eigs, target):
    
    eigs = np.asarray(eigs, dtype=complex)
    
    return np.min(
        np.abs(eigs - target)
        )

print("\n--- Eigenvalue Symmetry Validation ---")

for name in ["L1", "L2", "L3", "L4", "L5"]:
    
    position = getattr(points, name)
    
    x, y, z = position
    
    J = jacobian_J(
        x,
        y,
        z,
        mu
        )
    eigs = np.linalg.eigvals(J)

    minus_errors = []
    conjugate_errors = []
    minus_conjugate_errors = []

    for lam in eigs:

        minus_errors.append(
            nearest_eigenvalue_distance(
                eigs,
                -lam
                )
            )

        conjugate_errors.append(
            nearest_eigenvalue_distance(
                eigs,
                np.conj(lam)
            )
        )
    
        minus_conjugate_errors.append(
            nearest_eigenvalue_distance(
                eigs,
                -np.conj(lam)
            )
        )
        
        max_pm = np.max(minus_errors)
        max_conj = np.max(conjugate_errors)
        max_minus_conj = np.max(
            minus_conjugate_errors
            )
        
    print(f"\n{name}")
    
    print("Maximum +/- symmetry error:")
    print(np.max(minus_errors))
    
    print("Maximum conjugate symmetry error:")
    print(np.max(conjugate_errors))
    
    print("Maximum -conjugate symmetry error:")
    print(np.max(minus_conjugate_errors))
    
    assert max_pm < 1e-12
    assert max_conj < 1e-12
    assert max_minus_conj < 1e-12

print("\n--- Eigenvalue Structure ---")

for name in ["L1", "L2", "L3", "L4", "L5"]:

    position = getattr(points, name)

    x, y, z = position

    J = jacobian_J(
        x,
        y,
        z,
        mu
    )

    eigs = np.linalg.eigvals(J)

    eigs_sorted = sorted(
        eigs,
        key=lambda lam: (
            np.real(lam),
            np.imag(lam)
        )
    )

    print(f"\n{name}")

    for lam in eigs_sorted:

        print(
    f"{np.real(lam): .12e}  +  "
    f"{np.imag(lam): .12e} i"
         )

real_parts = np.real(eigs)

if name in ["L1", "L2", "L3"]:
    
    assert np.max(real_parts) > 1e-8
    assert np.min(real_parts) < 1e-8
    
else:
    
    assert np.max(
        np.abs(real_parts)
        ) < 1e-10

# ----------------------------------------------------
# Validation 10: Analytical L4/L5 Eigenfrequency

print("\n" + "=" * 60)
print("Validation 10 : Analytical L4/L5 Eigenfrequencies")
print("=" * 60)

D = np.sqrt(
    1.0 - 27.0 * mu * (1.0 - mu)
    )

omega_low = np.sqrt(
    (1.0 - D) / 2.0
    )

omega_high = np.sqrt(
    (1.0 + D) / 2.0
)

omega_vertical = 1.0

expected_frequencies = np.sort([
    omega_low,
    omega_high,
    omega_vertical
])

print("\nExpected analytical frequencies:")
print(expected_frequencies)

for name in ["L4", "L5"]:

    position = getattr(points, name)

    x, y, z = position

    J = jacobian_J(
        x,
        y,
        z,
        mu
    )

    eigs = np.linalg.eigvals(J)

    # Keep only positive imaginary members of each +/- pair
    numerical_frequencies = np.sort(
        np.imag(
            eigs[np.imag(eigs) > 0.0]
        )
    )

    error = np.abs(
        numerical_frequencies
        - expected_frequencies
    )

    max_error = np.max(error)

    print(f"\n{name}")

    print("Numerical frequencies:")
    print(numerical_frequencies)

    print("Individual frequency errors:")
    print(error)

    print("Maximum frequency error:")
    print(max_error)

    assert max_error < 1e-12

print("\nAnalytical L4/L5 frequency validation: PASS")

# ----------------------------------------------------
# Validation 11A: Jacobi / Effective Potential Consistancy

print("\n" + "=" * 60)
print("VALIDATION 11A: JACOBI / ZVC CONSISTENCY")
print("=" * 60)

for name in ["L1", "L2", "L3", "L4", "L5"]:

    position = getattr(points, name)

    x, y, z = position

    state = np.array([
        x,
        y,
        z,
        0.0,
        0.0,
        0.0
    ])

    C_jacobi = jacobi_constant(
        state,
        mu
    )

    C_omega = 2.0 * omega(
        x,
        y,
        z,
        mu
    )

    error = abs(
        C_jacobi - C_omega
    )

    print(f"\n{name}")
    print("C from Jacobi function:")
    print(C_jacobi)

    print("C from 2*Omega:")
    print(C_omega)

    print("Difference:")
    print(error)

    assert error < 1e-12

print("\nJacobi / effective-potential consistency: PASS")

# ----------------------------------------------------
# Validation 11B: Zero-Velocity COntour Residual

print("\n" + "=" * 60)
print("VALIDATION 11B: ZERO-VELOCITY CONTOUR")
print("=" * 60)

L1 = points.L1

C_L1 = 2.0 * omega(
    L1[0],
    L1[1],
    L1[2],
    mu
)

# Grid for contour calculation
x_vals = np.linspace(
    -1.5,
    1.5,
    800
)

y_vals = np.linspace(
    -1.5,
    1.5,
    800
)

X, Y = np.meshgrid(
    x_vals,
    y_vals
)

Z = np.full_like(
    X,
    np.nan,
    dtype=float
)

for j in range(X.shape[0]):

    for i in range(X.shape[1]):

        x = X[j, i]
        y = Y[j, i]

        r1, r2 = distances(
            x,
            y,
            0.0,
            mu
        )

        # Avoid primary singularities
        if r1 < 5e-3 or r2 < 5e-3:
            continue

        Z[j, i] = (
            2.0
            * omega(
                x,
                y,
                0.0,
                mu
            )
        )

# Generate contour without needing to display it
fig, ax = plt.subplots()

contour = ax.contour(
    X,
    Y,
    Z,
    levels=[C_L1]
)

# Collect contour vertices
vertices = []

for segment in contour.allsegs[0]:

    if len(segment) > 0:
        vertices.append(segment)

plt.close(fig)

if len(vertices) == 0:

    raise RuntimeError(
        "No C_L1 zero-velocity contour was found."
    )

vertices = np.vstack(vertices)

# Evaluate the actual ZVC equation at contour vertices
residuals = []

for x, y in vertices:

    residual = abs(
        2.0 * omega(
            x,
            y,
            0.0,
            mu
        )
        - C_L1
    )

    residuals.append(residual)

residuals = np.asarray(residuals)

print("\nC_L1:")
print(C_L1)

print("\nNumber of contour vertices:")
print(len(vertices))

print("\nMean ZVC residual:")
print(np.mean(residuals))

print("\nMaximum ZVC residual:")
print(np.max(residuals))

# ----------------------------------------------------
# Validation 12: Independent Lagrange-point solver comparison

print("\n" + "=" * 60)
print("VALIDATION 12: CROSS-MODULE LAGRANGE POINT AGREEMENT")
print("=" * 60)

points_bisection = lagrange_points_bisection(mu)

points_newton = lagrange_points_newton(mu)

max_position_error = 0.0

for name in ["L1", "L2", "L3", "L4", "L5"]:

    p_bisect = np.asarray(
        getattr(points_bisection, name),
        dtype=float
    )

    p_newton = np.asarray(
        points_newton[name],
        dtype=float
    )

    difference = (
        p_bisect - p_newton
    )

    error = np.linalg.norm(
        difference
    )

    max_position_error = max(
        max_position_error,
        error
    )

    print(f"\n{name}")

    print("Bisection result:")
    print(p_bisect)

    print("Newton result:")
    print(p_newton)

    print("Difference:")
    print(difference)

    print("Position error:")
    print(error)

    assert error < 1e-10

print("\nMaximum solver disagreement:")
print(max_position_error)

print(
    "\nIndependent Lagrange-point "
    "solver comparison: PASS"
)