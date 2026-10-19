import numpy as np

from crtbp.dynamics import jacobi_constant
from crtbp.lyapunov_corrector import (
    correct_planar_lyapunov
    )

def continue_lyapunov_family(
        seed_state,
        mu,
        dx,
        n_orbits,
        t_max,
        tol=1e-10,
        max_iterations=30,
        max_step=np.inf
        ):
    
    current_state = np.asarray(
        seed_state,
        dtype=float
        ).copy()
    
    family = []
    
    for i in range(n_orbits):
        
        result = correct_planar_lyapunov(
            state0=current_state,
            mu=mu,
            t_max=t_max,
            tol=tol,
            max_iterations=max_iterations,
            max_step=max_step
            )
        
        corrected_state = (
            result["state0"]
            )
        
        C = jacobi_constant(
            corrected_state,
            mu
            )
        
        family.append(
            {
                "index": i,
                "state0": corrected_state.copy(),
                "t_half": result["t_half"],
                "period": result["period"],
                "residual": result["residual"],
                "iterations": result["iterations"],
                "jacobi": C
                }
            )
        
        if i == n_orbits -1:
            break
        
        next_state = corrected_state.copy()
        next_state[0] += dx
        
        if len(family) >= 2:
            
            state_i = (family[-1]["state0"])
            state_im1 = (family[-2]["state0"])
            
            x_i = state_i[0]
            x_im1 = state_im1[0]
            
            vy_i = state_i[4]
            vy_im1 = state_im1[4]
            
            slope = (
                (vy_i - vy_im1) / (x_i - x_im1)
                )
            
            vy_predict = (vy_i + slope * dx)
            
            next_state[4] = (vy_predict)
            
        else:
            next_state[4] = (corrected_state[4])
        
        current_state = next_state
        
    return family