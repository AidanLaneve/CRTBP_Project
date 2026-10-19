# crtbp/equilibria.py
from __future__ import annotations
from dataclasses import dataclass
import math

from crtbp.constants import MU_EARTH_MOON

@dataclass(frozen=True)
class LagrangePoints:
    L1: tuple[float, float, float]
    L2: tuple[float, float, float]
    L3: tuple[float, float, float]
    L4: tuple[float, float, float]
    L5: tuple[float, float, float]

def _f_collinear(x: float, mu: float) -> float:
    # Primaries at (-mu, 0) and (1-mu, 0)
    r1 = abs(x + mu)
    r2 = abs(x - (1.0 - mu))
    return x - (1.0 - mu) * (x + mu) / (r1**3) - mu * (x - (1.0 - mu)) / (r2**3)

def _bisect_root(mu: float, a: float, b: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    fa = _f_collinear(a, mu)
    fb = _f_collinear(b, mu)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        raise ValueError(f"Bisection interval does not bracket a root: f(a)={fa}, f(b)={fb}")

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = _f_collinear(c, mu)
        if abs(fc) < tol or (b - a) < tol:
            return c
        # keep the sub-interval that brackets the root
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    raise RuntimeError("Bisection did not converge")

def lagrange_points(mu: float) -> LagrangePoints:
    """
    Returns (x,y,z) in normalized CRTBP rotating frame.
    mu = m2/(m1+m2), primaries at (-mu,0) and (1-mu,0).
    """
    if not (0.0 < mu < 0.5):
        raise ValueError("mu must be in (0, 0.5)")

    eps = 1e-6

    # Bracketing intervals
    # L1: between primaries
    a1 = -mu + eps
    b1 = (1.0 - mu) - eps

    # L2: right of smaller primary
    a2 = (1.0 - mu) + eps
    b2 = 2.0  # usually sufficient; increase if needed

    # L3: left of larger primary
    a3 = -2.0  # usually sufficient; decrease if needed
    b3 = -mu - eps

    x1 = _bisect_root(mu, a1, b1)
    x2 = _bisect_root(mu, a2, b2)
    x3 = _bisect_root(mu, a3, b3)

    # Triangular points (closed form)
    x4 = 0.5 - mu
    y4 = math.sqrt(3.0) / 2.0
    x5 = x4
    y5 = -y4

    return LagrangePoints(
        L1=(x1, 0.0, 0.0),
        L2=(x2, 0.0, 0.0),
        L3=(x3, 0.0, 0.0),
        L4=(x4, y4, 0.0),
        L5=(x5, y5, 0.0),
    )

if __name__ == "__main__":
    print(lagrange_points(MU_EARTH_MOON))