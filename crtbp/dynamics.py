# dynamics.py
import numpy as np

def crtbp_eom(t, s, mu):
    x, y, z, xd, yd, zd = s

    # positions relative to primaries (Earth at -mu, Moon at 1-mu)
    r1_vec = np.array([x + mu, y, z])
    r2_vec = np.array([x - (1 - mu), y, z])

    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    # effective potential partials (U_x, U_y, U_z) without the 1/2*(x^2+y^2) piece handled explicitly below
    # Common form: xdd - 2*yd = dΩ/dx, ydd + 2*xd = dΩ/dy, zdd = dΩ/dz
    # where Ω = 0.5*(x^2 + y^2) + (1-mu)/r1 + mu/r2
    dOmega_dx = x - (1 - mu) * (x + mu) / (r1**3) - mu * (x - (1 - mu)) / (r2**3)
    dOmega_dy = y - (1 - mu) * y / (r1**3) - mu * y / (r2**3)
    dOmega_dz =     - (1 - mu) * z / (r1**3) - mu * z / (r2**3)

    xdd =  2*yd + dOmega_dx
    ydd = -2*xd + dOmega_dy
    zdd = dOmega_dz

    return np.array([xd, yd, zd, xdd, ydd, zdd], dtype=float)

def jacobi_constant(s, mu):
    x, y, z, xd, yd, zd = s

    r1 = np.linalg.norm([x + mu, y, z])
    r2 = np.linalg.norm([x - (1 - mu), y, z])

    Omega = 0.5*(x**2 + y**2) + (1 - mu)/r1 + mu/r2
    v2 = xd*xd + yd*yd + zd*zd
    C = 2*Omega - v2
    return float(C)