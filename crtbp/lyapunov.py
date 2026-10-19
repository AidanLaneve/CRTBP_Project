import numpy as np

from crtbp.constants import MU_EARTH_MOON
from crtbp.equilibria import lagrange_points
from crtbp.linear_stability_analysis import (
    omega_partials_2nd,
    )

mu = MU_EARTH_MOON

def planar_frequency(xL, mu):
    
    Oxx, Oyy, Ozz, Oxy, Oxz, Oyz = omega_partials_2nd(
        xL, 
        0.0, 
        0.0, 
        mu
        )
    
    b = Oxx + Oyy - 4.0
    c = Oxx * Oyy
    
    discriminant = b**2 - 4.0 * c
    
    if discriminant < 0:
        raise ValueError("Planar frequency >0")
        
    sqrt_disc = np.sqrt(discriminant)
    
    q1 = (-b + sqrt_disc) / 2.0
    q2 = (-b - sqrt_disc) / 2.0
    
    positive_roots = [
        q for q in (q1, q2)
        if q > 0
        ]
    
    if len(positive_roots) == 0:
        raise ValueError("No positive oemga^2 root found")
        
    omega_squared = max(positive_roots)
    omega = np.sqrt(omega_squared)
    
    return omega, Oxx, Oyy

def linear_lyapunov_guess(mu, amplitude, point="L1"):
    
    if point not in ("L1", "L2", "L3"):
        raise ValueError("point must be a collinear Lagrange point")
        
    L = lagrange_points(mu)
    
    if point == "L1":
        xL = L.L1[0]       
    elif point == "L2":
        xL = L.L2[0]
    else: 
        xL = L.L3[0]
    
    omega, Oxx, Oyy = planar_frequency(xL, mu)
    
    A = amplitude
    x0 = xL + A
    vy0 = (-0.5 * (omega**2 + Oxx) * A)
    
    state0 = np.array(
        [x0,
         0.0,
         0.0,
         0.0,
         vy0,
         0.0
            ],
        dtype=float
        )
    T_linear = 2.0 * np.pi / omega
    
    return state0, omega, T_linear