import numpy as np

from scipy.integrate import solve_ivp

from crtbp.dynamics import crtbp_eom
from crtbp.linear_stability_analysis import jacobian_J

def augmented_eom (t, Y, mu):
    state = Y[:6]
    Phi = Y[6:].reshape(
        (6, 6)
        )
    
    state_dot = crtbp_eom(t, state, mu)
    
    x, y, z = state[:3]
    
    J = jacobian_J(x, y, z, mu)
    
    Phi_dot = J @ Phi
    
    return np.concatenate(
        (
        state_dot,
        Phi_dot.flatten()
        )
    )

def y_equals_zero_downward(t, Y):
    return Y[1]

y_equals_zero_downward.direction = -1
y_equals_zero_downward.terminal = True

def propagate_with_STM(
        state0,
        t_span,
        mu,
        rtol=1e-12,
        atol=1e-13,
        max_step=np.inf
        ):
    
    state0 = np.asarray(
        state0,
        dtype=float
        )
    
    Phi0 = np.eye(6)
    Y0 = np.concatenate(
        (
            state0,
            Phi0.flatten()
            )
        )
    
    sol = solve_ivp(
        fun=lambda t, Y:
            augmented_eom(
                t,
                Y,
                mu
                ),
        t_span=t_span,
        y0=Y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        events=y_equals_zero_downward,
        dense_output=True,
        max_step=max_step
        )
        
    return sol

def single_shooting_correction(
        state0,
        mu,
        t_max,
        damping=1.0,
        rtol=1e-12,
        atol=1e-13,
        max_step=np.inf
        ):
    
    state0 = np.asarray(
        state0,
        dtype=float
        )
    
    sol = propagate_with_STM(
        state0=state0,
        t_span=(0.0, t_max),
        mu=mu,
        rtol=rtol,
        atol=atol,
        max_step=max_step
        )
    
    if len(sol.t_events[0]) == 0:
        raise RuntimeError(
            "No downwards y=0 crossing detected."
            )
        
    t_half = sol.t_events[0][0]
    Yf = sol.y_events[0][0]
    
    state_f = Yf[:6]
    Phi_f = Yf[6:].reshape(
        (6, 6)
        )
    
    vx_f = state_f[3]
    vy_f = state_f[4]
    
    if abs(vy_f) < 1e-14:
        raise RuntimeError(
            "vy at the y=0 crossing is too small."
            )
    
    state_dot_f = crtbp_eom(
        t_half,
        state_f,
        mu
        )
    
    ax_f = state_dot_f[3]
    
    Phi_y_vy = Phi_f[1, 4]
    Phi_vx_vy = Phi_f[3, 4]
    
    D = (
        Phi_vx_vy - (ax_f / vy_f) * Phi_y_vy
        )
    
    if abs(D) < 1e-14:
        raise RuntimeError(
            "Differential correction denominator is too small."
            )
    
    raw_delta_vy0 = (-vx_f / D)
    applied_delta_vy0 = (damping * raw_delta_vy0)
    
    corrected_state0 = state0.copy()
    corrected_state0[4] += applied_delta_vy0
    
    return {
        "corrected_state0": corrected_state0,
        "raw_delta_vy0": raw_delta_vy0,
        "delta_vy0": applied_delta_vy0,
        "t_half": t_half,
        "state_f": state_f,
        "Phi_f": Phi_f,
        "vx_f": vx_f,
        "vy_f": vy_f,
        "ax_f": ax_f,
        "Phi_y_vy": Phi_y_vy,
        "Phi_vx_vy": Phi_vx_vy,
        "D": D
        }

def correct_planar_lyapunov(
        state0,
        mu,
        t_max,
        tol=1e-10,
        max_iterations=30,
        min_damping=1e-4,
        rtol=1e-12,
        atol=1e-13,
        max_step=np.inf
        ):
    
    current_state = np.asarray(
        state0,
        dtype=float
        ).copy()
    
    history = []
    
    for iteration in range(max_iterations):
        
        result = single_shooting_correction(
            state0=current_state,
            mu=mu,
            t_max=t_max,
            damping=1.0,
            rtol=rtol,
            atol=atol,
            max_step=max_step
            )
    
        residual = result["vx_f"]   
        residual_abs = abs(
            residual
            )
        
        print() 
        print(
            f"Iteration {iteration:2d}"
        )  
        print(
            f"    vy0       = "
            f"{current_state[4]: .12f}"
        )  
        print(
            f"    t_half    = "
            f"{result['t_half']:.12f}"
        )   
        print(
            f"    x_half    = "
            f"{result['state_f'][0]: .12f}"
        )    
        print(
            f"    vx_f      = "
            f"{residual: .12e}"
        )  
        print(
            f"    |vx_f|    = "
            f"{residual_abs:.12e}"
        )
    
        if residual_abs < tol:

            history.append(
                {
                    "iteration": iteration,
                    "vy0": current_state[4],
                    "t_half": result["t_half"],
                    "x_half": result["state_f"][0],
                    "vx_residual": residual,
                    "D": result["D"],
                    "raw_delta_vy0": 0.0,
                    "damping": 0.0,
                    "accepted_delta_vy0": 0.0
                }
            )

            return {
                "state0": current_state.copy(),
                "t_half": result["t_half"],
                "period": 2.0 * result["t_half"],
                "state_half": result["state_f"],
                "Phi_half": result["Phi_f"],
                "residual": residual,
                "iterations": iteration,
                "history": history
            }

        raw_delta = (
            result["raw_delta_vy0"]
        )

        print(
            f"    raw delta = "
            f"{raw_delta: .12e}"
        )

        alpha = 1.0

        accepted = False

        candidate_state = None
        candidate_result = None
        candidate_residual = None

        while alpha >= min_damping:

            candidate_state = (
                current_state.copy()
            )

            candidate_state[4] += (
                alpha
                * raw_delta
            )

            try:

                candidate_result = (
                    single_shooting_correction(
                        state0=candidate_state,
                        mu=mu,
                        t_max=t_max,
                        damping=1.0,
                        rtol=rtol,
                        atol=atol,
                        max_step=max_step
                    )
                )

            except RuntimeError:

                continue

            candidate_residual = abs(
                candidate_result["vx_f"]
            )

            if (
                candidate_residual
                <
                residual_abs
            ):

                accepted = True

                break

            alpha *= 0.5

        if not accepted:

            raise RuntimeError(
                "Backtracking failed to find "
                "an improving correction."
            )

        applied_delta = (
            alpha
            * raw_delta
        )

        print(
            f"    damping   = "
            f"{alpha:.6f}"
        )

        print(
            f"    applied   = "
            f"{applied_delta: .12e}"
        )

        print(
            f"    new |vx|  = "
            f"{candidate_residual:.12e}"
        )

        history.append(
            {
                "iteration": iteration,
                "vy0": current_state[4],
                "t_half": result["t_half"],
                "x_half": result["state_f"][0],
                "vx_residual": residual,
                "D": result["D"],
                "raw_delta_vy0": raw_delta,
                "damping": alpha,
                "accepted_delta_vy0": applied_delta,
                "candidate_vx_residual": candidate_result["vx_f"]
            }
        )

        current_state = (
            candidate_state
        )

    raise RuntimeError(
        "Planar Lyapunov differential correction "
        f"did not converge after "
        f"{max_iterations} iterations."
    )