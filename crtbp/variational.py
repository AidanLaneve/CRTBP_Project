import numpy as np

from scipy.integrate import solve_ivp

from crtbp.constants import MU_EARTH_MOON
from crtbp.dynamics import crtbp_eom
from crtbp.linear_stability_analysis import jacobian_J

mu = MU_EARTH_MOON


def stm_derivative(state, Phi, mu):
    
    x, y, z = state[:3]
    
    A = jacobian_J(x, y, z, mu)
    Phi_dot = A @ Phi
    
    return Phi_dot

def variational_eom(t, Y, mu):
    
    state = Y[:6]
    Phi = Y[6:].reshape((6, 6))
    
    state_dot = crtbp_eom(t, state, mu)
    Phi_dot = stm_derivative(state, Phi, mu)
    
    Phi_dot_flat = Phi_dot.flatten()
    
    Y_dot = np.concatenate((state_dot, Phi_dot_flat))
    
    return Y_dot

def propagate_variational(
        state0,
        t_span,
        mu,
        rtol=1e-11,
        atol=1e-13,
        dense_output=True,
        max_step=np.inf
        ):
    
    state0 = np.asarray(state0, dtype=float)
    Phi0 = np.eye(6)
    Y0 = np.concatenate((state0, Phi0.flatten()))
    
    def fun(t, Y):
        return variational_eom(t, Y, mu)
    
    sol = solve_ivp(
        fun=fun,
        t_span=t_span,
        y0=Y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=dense_output,
        max_step=max_step
        )
        
    return sol