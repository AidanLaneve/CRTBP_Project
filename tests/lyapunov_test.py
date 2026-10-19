import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from crtbp.constants import MU_EARTH_MOON
from crtbp.equilibria import lagrange_points
from crtbp.dynamics import jacobi_constant
from crtbp.propagate import (
    propagate,
    make_event_y_equals_0
    )
from crtbp.linear_stability_analysis import (
    jacobian_J
    )
from crtbp.lyapunov import (
    planar_frequency,
    linear_lyapunov_guess
    )
from crtbp.lyapunov_corrector import (
    single_shooting_correction,
    propagate_with_STM,
    correct_planar_lyapunov
    )
from crtbp.lyapunov_family import (
    continue_lyapunov_family
    )

mu = MU_EARTH_MOON

A = -0.01

state0, omega, T_linear = (
    linear_lyapunov_guess(
        mu=mu, 
        amplitude=A,
        point="L1"
        )
    )

L = lagrange_points(mu)

xL1 = L.L1[0]

omega_check, Oxx, Oyy = (
    planar_frequency(xL1, 
                     mu
                     )
                )

print()
print("=" * 60)
print("LINEAR L1 LYAPUNOV INITIAL GUESS")
print("=" * 60)

print(
    f"\nL1 x-coordinate = "
    f"{xL1:.12f}"
)

print(
    f"Amplitude A      = "
    f"{A:.12f}"
)

print("\nInitial state:")

names = [
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz"
]

for name, value in zip(
    names,
    state0
):

    print(
        f"{name:>2s} = "
        f"{value: .12f}"
    )

print(
    f"\nomega    = "
    f"{omega:.12f}"
)

print(
    f"T_linear = "
    f"{T_linear:.12f}"
)

# ---------------------------------
J = jacobian_J(
    xL1,
    0.0,
    0.0,
    mu
)

eigs = np.linalg.eigvals(J)

print("\nEigenvalues at L1:")

for eig in sorted(
    eigs,
    key=lambda z: (
        np.real(z),
        np.imag(z)
    )
):

    print(
        f"lambda = "
        f"{np.real(eig): .8f}"
        f"  +  "
        f"{np.imag(eig): .8f} i"
    )

sol = propagate(
    state0=state0,
    t_span=(0.0, T_linear),
    mu=mu,
    rtol=1e-12,
    atol=1e-13,
    dense_output=True,
    max_step=T_linear / 500
    )

y_cross_event = make_event_y_equals_0(
    direction=-1
    )

sol_half = propagate(
    state0=state0,
    t_span=(0.0, 2.0 * T_linear),
    mu=mu,
    rtol=1e-12,
    atol=1e-13,
    events=y_cross_event,
    dense_output=True,
    max_step=T_linear / 500
    )

if len(sol_half.t_events[0]) == 0:
    raise RuntimeError(
        "No downward y=0 crossings"
        )
    
t_half = sol_half.t_events[0][0]
state_half = sol_half.y_events[0][0]

(
 x_half,
 y_half,
 z_half,
 vx_half,
 vy_half,
 vz_half,
 ) = state_half

print()
print("=" * 60)
print("Half-Period Shooting Test")
print("=" * 60)

print(
      f"\nDetected crossing time = "
      f"{t_half:.12f}"
      )
print(
      f"Linear half period = "
      f"{0.5 * T_linear:.12f}"
      )
print("\nCrossing state:")
for name, value in zip(
        names,
        state_half
        ):
    print(
        f"{name:>2s} = "
        f"{value: .12f}"
        )
print(
    f"\nHalf-period residual vx = "
    f"{vx_half:.12e}"
    )
half_period_error = (
    t_half - 0.5 * T_linear
    )
print(
    f"Half-period timing error = "
    f"{half_period_error:.12e}"
    )

t_half_plot = np.linspace(
    0.0,
    t_half,
    250
    )
states_half_plot = sol_half.sol(
    t_half_plot
    )

x_half_plot = states_half_plot[0]
y_half_plot = states_half_plot[1]


# ================================================

# Figure 1 : Animated
n_frames = 400

t_anim = np.linspace(
    0.0,
    T_linear,
    n_frames
    )

states_anim = sol.sol(t_anim)

x_anim = states_anim[0]
y_anim = states_anim[1]

kappa = (omega**2 + Oxx) / (2.0 * omega)

x_linear = (xL1 + A * np.cos(omega * t_anim))
y_linear = (-kappa * A * np.sin(omega * t_anim))

fig_anim, ax_anim = plt.subplots(figsize=(8, 6))

ax_anim.scatter(
    -mu,
    0.0,
    s=80,
    label="Earth"
    )
ax_anim.scatter(
    1.0 - mu,
    0.0,
    s=50,
    label="Moon"
    )
ax_anim.scatter(
    xL1,
    0.0,
    marker="x",
    s=100,
    label="L1"
    )
ax_anim.plot(
    x_linear,
    y_linear,
    linestyle="--",
    linewidth=1.5,
    label="Linear prediction"
    )
trail, = ax_anim.plot(
    [],
    [],
    linewidth=2,
    label="Nonlinear trajectory"
)

# Spacecraft marker
spacecraft, = ax_anim.plot(
    [],
    [],
    marker="o",
    markersize=8,
    linestyle="None"
)

# Time display
time_text = ax_anim.text(
    0.03,
    0.95,
    "",
    transform=ax_anim.transAxes
)

# ------------------------------------------------------------
# Axis limits
# ------------------------------------------------------------

margin_x = 0.02
margin_y = 0.02

xmin = min(
    np.min(x_anim),
    np.min(x_linear)
) - margin_x

xmax = max(
    np.max(x_anim),
    np.max(x_linear)
) + margin_x

ymin = min(
    np.min(y_anim),
    np.min(y_linear)
) - margin_y

ymax = max(
    np.max(y_anim),
    np.max(y_linear)
) + margin_y

ax_anim.set_xlim(
    xmin,
    xmax
)

ax_anim.set_ylim(
    ymin,
    ymax
)

ax_anim.set_aspect(
    "equal",
    adjustable="box"
)

ax_anim.set_xlabel("x")
ax_anim.set_ylabel("y")

ax_anim.set_title(
    "L1 Lyapunov Initial Guess\n"
    "Nonlinear CR3BP Propagation"
)

ax_anim.grid(True)
ax_anim.legend()

def init_animation():

    trail.set_data(
        [],
        []
    )

    spacecraft.set_data(
        [],
        []
    )

    time_text.set_text("")

    return (
        trail,
        spacecraft,
        time_text
    )
def update_animation(frame):

    trail.set_data(
        x_anim[:frame + 1],
        y_anim[:frame + 1]
    )

    spacecraft.set_data(
        [x_anim[frame]],
        [y_anim[frame]]
    )

    time_text.set_text(
        f"t = {t_anim[frame]:.3f}"
    )

    return (
        trail,
        spacecraft,
        time_text
    )

animation = FuncAnimation(
    fig_anim,
    update_animation,
    frames=n_frames,
    init_func=init_animation,
    interval=20,
    blit=True,
    repeat=True
)

# Figure 2 : Static

x = sol.y[0]
y = sol.y[1]

fig_static, ax_static = (
    plt.subplots(figsize=(8, 6))
    )

ax_static.plot(
    x,
    y,
    linewidth=2,
    label="Nonlinear propagation"
    )

ax_static.plot(
    x_linear,
    y_linear,
    linestyle="--",
    linewidth=1.5,
    label="Linear prediction"
    )

ax_static.scatter(
    xL1,
    0.0,
    marker="x",
    s=100,
    label="l1"
    )

ax_static.scatter(
    state0[0],
    state0[1],
    marker="o",
    s=50,
    label="Initial state"
    )

ax_static.scatter(
    -mu,
    0.0,
    marker="o",
    s=80,
    label="Earth"
    )

ax_static.scatter(
    1.0 - mu,
    0.0,
    marker="o",
    s=50,
    label="Moon"
    )

ax_static.set_title(
    "Uncorrected Linear L1 Lyapunov Guess\n"
    "Propagated in Full Nonlinear CRTBP"
    )

ax_static.axis("equal")
ax_static.grid(True)
ax_static.legend()

# Figure 3 : Half period shooting geometry

fig_half, ax_half = plt.subplots(
    figsize=(8, 6)
    )

ax_half.plot(
    x_half_plot,
    y_half_plot,
    linewidth=2,
    label="Trajectory to next y=0 crossing"
    )

ax_half.scatter(
    state0[0],
    state0[1],
    s=60,
    label="Initial state"
    )

ax_half.scatter(
    x_half,
    y_half,
    s=60,
    label="Half-period crossing"
    )

ax_half.scatter(
    xL1,
    0.0,
    marker="x",
    s=100,
    label="L1"
    )

ax_half.axhline(
    0.0,
    linewidth=1,
    linestyle="--"
    )

ax_half.set_title("Half period shooting geometry")

ax_half.axis("equal")
ax_half.grid(True)
ax_half.legend()


# lyapunov corrector test
result = single_shooting_correction(
    state0=state0,
    mu=mu,
    t_max=2.0 * T_linear,
    max_step=T_linear / 500
    )
# ----------------

print()
print('=' * 60)
print("Single Differential Correction")
print('=' * 60)

print(
      f"\nOld vy0 = "
      f"{state0[4]: .12f}"
      )

print(
      f"vx at crossing = "
      f"{result['vx_f']: .12e}"
      )
print(
      f"vy at crossing = "
      f"{result['vy_f']: .12e}"
      )
print(
      f"ax at crossing = "
      f"{result['ax_f']: .12e}"
      )

print(
      f"\nPhi[1,4] = "
      f"{result['Phi_y_vy']: .12e}"
      )
print(
      f"Phi[3,4] = "
      f"{result['Phi_vx_vy']: .12e}"
      )

print(
      f"\nCorrection derivative D = "
      f"{result['D']: .12e}"
      )

print(
      f"Delta vy0 = "
      f"{result['delta_vy0']: .12e}"
      )
print(
      f"New vy0 = "
      f"{result['corrected_state0'][4]: .12e}"
      )


corrected_state0 = result["corrected_state0"]

sol_corrected = propagate_with_STM(
    state0=corrected_state0,
    t_span=(0.0, 2.0 * T_linear),
    mu=mu,
    rtol=1e-12,
    atol=1e-13,
    max_step=T_linear / 500
    )

if len(sol_corrected.t_events[0]) == 0:
    raise RuntimeError(
        "Corrected trajectory did not produce downward y=0 crossing."
        )
    
t_half_corrected = (
    sol_corrected.t_events[0][0]
    )
Yf_corrected = (
    sol_corrected.y_events[0][0]
    )
state_f_corrected = (
    Yf_corrected[:6]
    )
vx_f_corrected = (
    state_f_corrected[3]
    )
vy_f_corrected = (
    state_f_corrected[4]
    )

print()
print("=" * 60)
print("FIRST CORRECTION VALIDATION")
print("=" * 60)

print(
    f"\nOriginal vy0   = "
    f"{state0[4]: .12f}"
)
print(
    f"Corrected vy0  = "
    f"{corrected_state0[4]: .12f}"
)

print(
    f"\nOriginal crossing time  = "
    f"{result['t_half']:.12f}"
)
print(
    f"Corrected crossing time = "
    f"{t_half_corrected:.12f}"
)

print(
    f"\nOriginal vx residual  = "
    f"{result['vx_f']: .12e}"
)
print(
    f"Corrected vx residual = "
    f"{vx_f_corrected: .12e}"
)

print(
    f"\nOriginal |vx|  = "
    f"{abs(result['vx_f']):.12e}"
)
print(
    f"Corrected |vx| = "
    f"{abs(vx_f_corrected):.12e}"
)

print(
    f"\nCorrected crossing x  = "
    f"{state_f_corrected[0]:.12f}"
)
print(
    f"Corrected crossing vy = "
    f"{vy_f_corrected:.12f}"
)

# corrected
print()
print("=" * 60)
print("AUTOMATIC DIFFERENTIAL CORRECTION")
print("=" * 60)

corrected = correct_planar_lyapunov(
    state0=state0,
    mu=mu,
    t_max=2.0 * T_linear,
    tol=1e-10,
    max_iterations=20,
    max_step=T_linear / 500
)

corrected_state0 = corrected["state0"]

print()
print("=" * 60)
print("CORRECTED LYAPUNOV ORBIT")
print("=" * 60)

print("\nCorrected initial state:")

for name, value in zip(
    names,
    corrected_state0
):
    print(
        f"{name:>2s} = "
        f"{value: .12f}"
    )

print(
    f"\nHalf-period = "
    f"{corrected['t_half']:.12f}"
)
print(
    f"Full period = "
    f"{corrected['period']:.12f}"
)
print(
    f"Final vx residual = "
    f"{corrected['residual']:.12e}"
)
print(
    f"Iterations = "
    f"{corrected['iterations']}"
)

# full-period validation

T_corrected = corrected["period"]
corrected_state0 = corrected["state0"]

sol_corrected_full = propagate(
    state0=corrected_state0,
    t_span=(0.0, T_corrected),
    mu=mu,
    rtol=1e-12,
    atol=1e-13,
    dense_output=True,
    max_step=T_corrected / 500
    )

state_corrected_final = sol_corrected_full.sol(T_corrected)

closure_error_corrected = (
    state_corrected_final - corrected_state0
    )
closure_norm_corrected = np.linalg.norm(
    closure_error_corrected
    )

C0_corrected = jacobi_constant(
    corrected_state0,
    mu
    )
Cf_corrected = jacobi_constant(
    state_corrected_final,
    mu
    )

jacobi_drift_corrected = (
    Cf_corrected
    - C0_corrected
)

print()
print("=" * 60)
print("FULL-PERIOD VALIDATION OF CORRECTED ORBIT")
print("=" * 60)

print("\nCorrected final state:")

for name, value in zip(
    names,
    state_corrected_final
    ):
    
    print(
        f"{name:>2s} = "
        f"{value: .12f}"
    )

print("\nCorrected closure error:")

for name, value in zip(
    names,
    closure_error_corrected
    ):
    
    print(
        f"d{name:>2s} = "
        f"{value: .12e}"
    )

print(
    "\nCorrected state closure norm = "
    f"{closure_norm_corrected:.12e}"
)
print(
    f"Initial Jacobi constant = "
    f"{C0_corrected:.15f}"
)
print(
    f"Final Jacobi constant   = "
    f"{Cf_corrected:.15f}"
)
print(
    f"Jacobi drift            = "
    f"{jacobi_drift_corrected:.12e}"
)

# Figure 4 : Animated Corrected vs Uncorrected Trajectories

t_corrected_anim = np.linspace(
    0.0,
    T_corrected,
    n_frames
    )

states_corrected_anim = sol_corrected_full.sol(
    t_corrected_anim
    )

x_corrected_anim = states_corrected_anim[0]
y_corrected_anim = states_corrected_anim[1]

fig_compare, ax_compare = plt.subplots(
    figsize=(8, 6)
    )

ax_compare.scatter(
    -mu,
    0.0,
    s=80,
    label="Earth"
)
ax_compare.scatter(
    1.0 - mu,
    0.0,
    s=50,
    label="Moon"
)
ax_compare.scatter(
    xL1,
    0.0,
    marker="x",
    s=100,
    label="L1"
)

ax_compare.plot(
    x_anim,
    y_anim,
    linestyle="--",
    linewidth=1.0,
    alpha=0.5,
    label="Uncorrected full path"
)
ax_compare.plot(
    x_corrected_anim,
    y_corrected_anim,
    linestyle="--",
    linewidth=1.0,
    alpha=0.5,
    label="Corrected full path"
)

trail_uncorrected, = ax_compare.plot(
    [],
    [],
    linewidth=2,
    label="Uncorrected trajectory"
)
trail_corrected, = ax_compare.plot(
    [],
    [],
    linewidth=2,
    label="Corrected trajectory"
)

marker_uncorrected, = ax_compare.plot(
    [],
    [],
    marker="o",
    markersize=8,
    linestyle="None"
)
marker_corrected, = ax_compare.plot(
    [],
    [],
    marker="s",
    markersize=8,
    linestyle="None"
)

phase_text = ax_compare.text(
    0.03,
    0.95,
    "",
    transform=ax_compare.transAxes
)

margin_x = 0.02
margin_y = 0.02

xmin_compare = min(
    np.min(x_anim),
    np.min(x_corrected_anim)
) - margin_x

xmax_compare = max(
    np.max(x_anim),
    np.max(x_corrected_anim)
) + margin_x

ymin_compare = min(
    np.min(y_anim),
    np.min(y_corrected_anim)
) - margin_y

ymax_compare = max(
    np.max(y_anim),
    np.max(y_corrected_anim)
) + margin_y

ax_compare.set_xlim(
    xmin_compare,
    xmax_compare
)

ax_compare.set_ylim(
    ymin_compare,
    ymax_compare
)

ax_compare.set_aspect(
    "equal",
    adjustable="box"
)

ax_compare.set_xlabel("x")
ax_compare.set_ylabel("y")

ax_compare.set_title(
    "Uncorrected vs Corrected Lyapunov Orbit\n"
    "Animated Comparison"
)

ax_compare.grid(True)
ax_compare.legend()

def init_compare_animation():

    trail_uncorrected.set_data([], [])
    trail_corrected.set_data([], [])

    marker_uncorrected.set_data([], [])
    marker_corrected.set_data([], [])

    phase_text.set_text("")

    return (
        trail_uncorrected,
        trail_corrected,
        marker_uncorrected,
        marker_corrected,
        phase_text
    )


def update_compare_animation(frame):

    # Normalized phase from 0 to 1
    phase = frame / (n_frames - 1)

    # Uncorrected
    trail_uncorrected.set_data(
        x_anim[:frame + 1],
        y_anim[:frame + 1]
    )

    marker_uncorrected.set_data(
        [x_anim[frame]],
        [y_anim[frame]]
    )

    trail_corrected.set_data(
        x_corrected_anim[:frame + 1],
        y_corrected_anim[:frame + 1]
    )

    marker_corrected.set_data(
        [x_corrected_anim[frame]],
        [y_corrected_anim[frame]]
    )

    phase_text.set_text(
        f"phase = {phase:.3f}\n"
        f"t_uncorr = {t_anim[frame]:.3f}\n"
        f"t_corr = {t_corrected_anim[frame]:.3f}"
    )

    return (
        trail_uncorrected,
        trail_corrected,
        marker_uncorrected,
        marker_corrected,
        phase_text
    )


animation_compare = FuncAnimation(
    fig_compare,
    update_compare_animation,
    frames=n_frames,
    init_func=init_compare_animation,
    interval=20,
    blit=True,
    repeat=True
)

print()
print("=" * 60)
print("LYAPUNOV FAMILY CONTINUATION")
print("=" * 60)

family = continue_lyapunov_family(
    seed_state=corrected_state0,
    mu=mu,
    dx=-0.001,
    n_orbits=5,
    t_max=2.0 * T_linear,
    tol=1e-10,
    max_iterations=30,
    max_step=T_linear / 500
    )

print()
for orbit in family:
    
    state = orbit["state0"]
    
    print(
        f"Orbit {orbit['index']:2d}: "
        f"x0 = {state[0]: .12f}, "
        f"vy0 = {state[4]: .12f}, "
        f"T = {orbit['period']:.12f}, "
        f"C = {orbit['jacobi']:.12f}, "
        f"residual = {orbit['residual']:.3e}"
        )

# Figure 5 : Lyapunov Orbit Family

fig_family, ax_family = plt.subplots(
    figsize=(8, 6)
    )

colors = plt.cm.viridis(
    np.linspace(0, 1, len(family))
    )

for i, orbit in enumerate(family):
    
    state_i = orbit["state0"]
    T_i = orbit["period"]
    sol_i = propagate(
        state0=state_i,
        t_span=(0.0, T_i),
        mu=mu,
        rtol=1e-12,
        atol=1e-13,
        dense_output=True,
        max_step=T_i / 500
        )
    
    t_i = np.linspace(
        0.0,
        T_i,
        500
        )
    
    states_i = sol_i.sol(t_i)
    
    final_state = sol_i.sol(T_i)
    
    closure_error = np.linalg.norm(
        final_state - state_i
        )
    
    jacobi_drift = (
        jacobi_constant(final_state, mu)
        - jacobi_constant(state_i, mu)
        )
    
    print(
        f"Orbit {orbit['index']}: "
        f"Closure = {closure_error:.3e}, "
        f"Jacobi drift = {jacobi_drift:.3e}"
    )
    
    x_i = states_i[0]
    y_i = states_i[1]
    
    ax_family.plot(
        x_i,
        y_i,
        linewidth=1.5,
        color=colors[i],
        label=(
            f"Orbit {orbit['index']} "
            f"(C={orbit['jacobi']:.6f})"
            )
        )
    
ax_family.scatter(
        -mu,
        0.0,
        s=80,
        label="Earth"
        )
ax_family.scatter(
        1.0 - mu,
        0.0,
        s=50,
        label="Moon"
        )
ax_family.scatter(
        xL1,
        0.0,
        marker="x",
        s=100,
        label="L1"
        )
    
ax_family.set_title("Planar L1 Lyapunov Orbit Family")

ax_family.set_xlim(
    xL1 - 0.05,
    xL1+ 0.05
    )
ax_family.set_ylim(
    -0.06,
    0.06
    )
ax_family.set_aspect(
    "equal",
    adjustable="box"
    )

ax_family.grid(True)
ax_family.legend()


fig_anim.tight_layout()
fig_static.tight_layout()
fig_half.tight_layout()
fig_compare.tight_layout()
fig_family.tight_layout()

plt.show()