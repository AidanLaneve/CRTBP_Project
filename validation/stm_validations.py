import numpy as np

from crtbp.constants import MU_EARTH_MOON
from crtbp.propagate import propagate

from crtbp.variational import propagate_variational

mu = MU_EARTH_MOON


if __name__ == "__main__":

    state0 = np.array([
        0.8,
        0.1,
        0.0,
        0.0,
        0.2,
        0.0
    ])

    # ============================================
    # VARIATIONAL PROPAGATION
    # ============================================

    sol = propagate_variational(
        state0,
        (0.0, 1.0),
        mu
    )

    print("\nIntegration successful:")
    print(sol.success)

    print("\nSolver message:")
    print(sol.message)

    print("\nSolution array shape:")
    print(sol.y.shape)

    state_final = sol.y[:6, -1]

    print("\nInitial state:")
    print(state0)

    print("\nFinal state:")
    print(state_final)

    Phi_final = sol.y[6:, -1].reshape((6, 6))

    print("\nInitial STM:")
    print(np.eye(6))

    print("\nFinal STM:")
    print(Phi_final)


    # ============================================
    # NOMINAL TRAJECTORY VALIDATION
    # ============================================

    sol_normal = propagate(
        state0,
        (0.0, 1.0),
        mu
    )

    state_final_normal = sol_normal.y[:6, -1]

    state_final_variational = sol.y[:6, -1]

    difference = (
        state_final_variational
        - state_final_normal
    )

    print("\n" + "=" * 60)
    print("NOMINAL TRAJECTORY VALIDATION")
    print("=" * 60)

    print("\nFinal state from ordinary propagator:")
    print(state_final_normal)

    print("\nFinal state from variational propagator:")
    print(state_final_variational)

    print("\nDifference:")
    print(difference)

    print("\nMaximum absolute difference:")
    print(np.max(np.abs(difference)))
    
    # ========================================================
# FINITE-DIFFERENCE VALIDATION
# FIRST STM COLUMN: dXf / dx0
# ========================================================

print("\n" + "=" * 60)
print("FINITE-DIFFERENCE STM VALIDATION")
print("Testing first column: dXf / dx0")
print("=" * 60)

# Size of initial x perturbation
epsilon = 1e-6


# --------------------------------------------------------
# Create perturbed initial conditions
# --------------------------------------------------------

state_plus = state0.copy()
state_minus = state0.copy()

state_plus[0] += epsilon
state_minus[0] -= epsilon


print("\nOriginal initial state:")
print(state0)

print("\nPositive perturbation:")
print(state_plus)

print("\nNegative perturbation:")
print(state_minus)


# --------------------------------------------------------
# Propagate both perturbed trajectories
# --------------------------------------------------------

sol_plus = propagate(
    state_plus,
    (0.0, 1.0),
    mu
)

sol_minus = propagate(
    state_minus,
    (0.0, 1.0),
    mu
)


# --------------------------------------------------------
# Extract final states
# --------------------------------------------------------

state_final_plus = sol_plus.y[:6, -1]
state_final_minus = sol_minus.y[:6, -1]


print("\nFinal state from +epsilon:")
print(state_final_plus)

print("\nFinal state from -epsilon:")
print(state_final_minus)


# --------------------------------------------------------
# Central finite difference
# --------------------------------------------------------

fd_column = (
    state_final_plus
    - state_final_minus
) / (2 * epsilon)


# --------------------------------------------------------
# First column from propagated STM
# --------------------------------------------------------

stm_column = Phi_final[:, 0]


# --------------------------------------------------------
# Compare
# --------------------------------------------------------

column_difference = (
    stm_column
    - fd_column
)


print("\nSTM prediction:")
print(stm_column)

print("\nFinite-difference approximation:")
print(fd_column)

print("\nDifference:")
print(column_difference)

print("\nMaximum absolute difference:")
print(
    np.max(
        np.abs(column_difference)
    )
)
    # ========================================================
    # FULL FINITE-DIFFERENCE STM VALIDATION
    # ========================================================

print("\n" + "=" * 60)
print("FULL FINITE-DIFFERENCE STM VALIDATION")
print("=" * 60)

epsilon = 1e-6

# Empty 6x6 matrix that will hold the
# finite-difference approximation to the STM
Phi_fd = np.zeros((6, 6))


# --------------------------------------------------------
# Perturb each initial state variable
# --------------------------------------------------------

for j in range(6):

    state_plus = state0.copy()
    state_minus = state0.copy()

    state_plus[j] += epsilon
    state_minus[j] -= epsilon


    # Propagate positive perturbation
    sol_plus = propagate(
        state_plus,
        (0.0, 1.0),
        mu
    )


    # Propagate negative perturbation
    sol_minus = propagate(
        state_minus,
        (0.0, 1.0),
        mu
    )


    # Final states
    Xf_plus = sol_plus.y[:6, -1]
    Xf_minus = sol_minus.y[:6, -1]


    # Central finite difference
    Phi_fd[:, j] = (
        Xf_plus
        - Xf_minus
    ) / (2 * epsilon)


# --------------------------------------------------------
# Compare propagated STM with finite-difference STM
# --------------------------------------------------------

Phi_error = Phi_final - Phi_fd


print("\nPropagated STM:")
print(Phi_final)

print("\nFinite-difference STM:")
print(Phi_fd)

print("\nDifference:")
print(Phi_error)

print("\nMaximum absolute difference:")
print(
    np.max(
        np.abs(Phi_error)
    )
)
# ========================================================
# FINITE-DIFFERENCE EPSILON CONVERGENCE STUDY
# ========================================================

print("\n" + "=" * 60)
print("FINITE-DIFFERENCE EPSILON CONVERGENCE STUDY")
print("=" * 60)

# Perturbation sizes to test
epsilons = [
    1e-3,
    1e-4,
    1e-5,
    1e-6,
    1e-7,
    1e-8,
    1e-9
]

# Store the maximum error for each epsilon
errors = []


# --------------------------------------------------------
# Loop over perturbation sizes
# --------------------------------------------------------

for epsilon in epsilons:

    # Empty finite-difference STM
    Phi_fd = np.zeros((6, 6))


    # ----------------------------------------------------
    # Construct all six columns
    # ----------------------------------------------------

    for j in range(6):

        state_plus = state0.copy()
        state_minus = state0.copy()

        state_plus[j] += epsilon
        state_minus[j] -= epsilon


        # Positive perturbation
        sol_plus = propagate(
            state_plus,
            (0.0, 1.0),
            mu
        )


        # Negative perturbation
        sol_minus = propagate(
            state_minus,
            (0.0, 1.0),
            mu
        )


        # Final states
        Xf_plus = sol_plus.y[:6, -1]
        Xf_minus = sol_minus.y[:6, -1]


        # Central finite difference
        Phi_fd[:, j] = (
            Xf_plus
            - Xf_minus
        ) / (2 * epsilon)


    # ----------------------------------------------------
    # Compare finite-difference STM to propagated STM
    # ----------------------------------------------------

    Phi_error = (
        Phi_final
        - Phi_fd
    )

    max_error = np.max(
        np.abs(Phi_error)
    )

    errors.append(max_error)


    print(
        f"epsilon = {epsilon:.1e}"
        f"    max error = {max_error:.6e}"
    )


# --------------------------------------------------------
# Find best epsilon
# --------------------------------------------------------

best_index = np.argmin(errors)

best_epsilon = epsilons[best_index]
best_error = errors[best_index]


print("\nBest epsilon:")
print(best_epsilon)

print("\nMinimum maximum error:")
print(best_error)
# ========================================================
# STM COMPOSITION PROPERTY VALIDATION
# ========================================================

print("\n" + "=" * 60)
print("STM COMPOSITION PROPERTY VALIDATION")
print("=" * 60)

# Times
t0 = 0.0
t1 = 0.5
t2 = 1.0


# --------------------------------------------------------
# Propagate directly from t0 to t2
# --------------------------------------------------------

sol_02 = propagate_variational(
    state0,
    (t0, t2),
    mu
)

Phi_02 = (
    sol_02.y[6:, -1]
    .reshape((6, 6))
)


# --------------------------------------------------------
# Propagate from t0 to t1
# --------------------------------------------------------

sol_01 = propagate_variational(
    state0,
    (t0, t1),
    mu
)

state_t1 = sol_01.y[:6, -1]

Phi_01 = (
    sol_01.y[6:, -1]
    .reshape((6, 6))
)


# --------------------------------------------------------
# Propagate from t1 to t2
#
# The STM is automatically initialized to I,
# so this gives Phi(t2, t1).
# --------------------------------------------------------

sol_12 = propagate_variational(
    state_t1,
    (t1, t2),
    mu
)

Phi_12 = (
    sol_12.y[6:, -1]
    .reshape((6, 6))
)


# --------------------------------------------------------
# Composition
# --------------------------------------------------------

Phi_composed = Phi_12 @ Phi_01


# --------------------------------------------------------
# Compare
# --------------------------------------------------------

composition_error = (
    Phi_02
    - Phi_composed
)


print("\nDirect STM Phi(t2, t0):")
print(Phi_02)

print("\nComposed STM Phi(t2,t1) @ Phi(t1,t0):")
print(Phi_composed)

print("\nDifference:")
print(composition_error)

print("\nMaximum absolute difference:")
print(
    np.max(
        np.abs(composition_error)
    )
)