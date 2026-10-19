# ============================================================
# CR3BP LAGRANGE POINT LINEAR STABILITY ANALYSIS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from crtbp.constants import MU_EARTH_MOON


mu = MU_EARTH_MOON

def distances(x, y, z, mu):


    r1 = np.sqrt(
        (x + mu)**2
        + y**2
        + z**2
    )

    r2 = np.sqrt(
        (x - (1 - mu))**2
        + y**2
        + z**2
    )

    return r1, r2


def omega(x, y, z, mu):

    r1, r2 = distances(x, y, z, mu)

    return (
        0.5 * (x**2 + y**2)
        + (1 - mu) / r1
        + mu / r2
    )

def omega_partials_2nd(x, y, z, mu):

    r1, r2 = distances(x, y, z, mu)

    eps = 1e-14

    r1 = max(r1, eps)
    r2 = max(r2, eps)

    mu1 = 1 - mu

    a = x + mu
    b = x - (1 - mu)

    r1_3 = r1**3
    r1_5 = r1**5

    r2_3 = r2**3
    r2_5 = r2**5

    Oxx = (
        1
        - mu1 * (1 / r1_3 - 3 * a**2 / r1_5)
        - mu  * (1 / r2_3 - 3 * b**2 / r2_5)
    )

    Oyy = (
        1
        - mu1 * (1 / r1_3 - 3 * y**2 / r1_5)
        - mu  * (1 / r2_3 - 3 * y**2 / r2_5)
    )

    Ozz = (
        - mu1 * (1 / r1_3 - 3 * z**2 / r1_5)
        - mu  * (1 / r2_3 - 3 * z**2 / r2_5)
    )

    Oxy = 3 * (
        mu1 * a * y / r1_5
        + mu * b * y / r2_5
    )

    Oxz = 3 * (
        mu1 * a * z / r1_5
        + mu * b * z / r2_5
    )

    Oyz = 3 * (
        mu1 * y * z / r1_5
        + mu * y * z / r2_5
    )

    return Oxx, Oyy, Ozz, Oxy, Oxz, Oyz


# ============================================================
# LINEARIZED CR3BP JACOBIAN
# ============================================================

def jacobian_J(x, y, z, mu):

    (
        Oxx,
        Oyy,
        Ozz,
        Oxy,
        Oxz,
        Oyz
    ) = omega_partials_2nd(x, y, z, mu)

    J = np.zeros((6, 6), dtype=float)

    J[0, 3] = 1
    J[1, 4] = 1
    J[2, 5] = 1

    J[3, 0] = Oxx
    J[3, 1] = Oxy
    J[3, 2] = Oxz

    J[3, 4] = 2

    J[4, 0] = Oxy
    J[4, 1] = Oyy
    J[4, 2] = Oyz

    J[4, 3] = -2

    J[5, 0] = Oxz
    J[5, 1] = Oyz
    J[5, 2] = Ozz

    return J

def collinear_root(mu, which, tol=1e-13, maxiter=100):

    def dOmega_dx(x):

        r1 = abs(x + mu)
        r2 = abs(x - (1 - mu))

        return (
            x
            - (1 - mu) * (x + mu) / r1**3
            - mu * (x - (1 - mu)) / r2**3
        )

    def d2Omega_dx2(x):

        r1 = abs(x + mu)
        r2 = abs(x - (1 - mu))

        return (
            1
            - (1 - mu)
            * (
                1 / r1**3
                - 3 * (x + mu)**2 / r1**5
            )
            - mu
            * (
                1 / r2**3
                - 3 * (x - (1 - mu))**2 / r2**5
            )
        )

    if which == "L1":

        x = (
            1
            - mu
            - (mu / 3)**(1 / 3)
        )

    elif which == "L2":

        x = (
            1
            - mu
            + (mu / 3)**(1 / 3)
        )

    elif which == "L3":

        x = -1 - (5 * mu / 12)

    else:

        raise ValueError(
            "which must be 'L1', 'L2', or 'L3'"
        )

    for _ in range(maxiter):

        f = dOmega_dx(x)
        fp = d2Omega_dx2(x)

        step = f / fp

        x_new = x - step

        if abs(x_new - x) < tol:
            return x_new

        x = x_new

    raise RuntimeError(
        f"Newton method did not converge for {which}"
    )

def lagrange_points(mu):

    L1 = (
        collinear_root(mu, "L1"),
        0.0,
        0.0
    )

    L2 = (
        collinear_root(mu, "L2"),
        0.0,
        0.0
    )

    L3 = (
        collinear_root(mu, "L3"),
        0.0,
        0.0
    )

    L4 = (
        0.5 - mu,
        np.sqrt(3) / 2,
        0.0
    )

    L5 = (
        0.5 - mu,
        -np.sqrt(3) / 2,
        0.0
    )

    return {
        "L1": L1,
        "L2": L2,
        "L3": L3,
        "L4": L4,
        "L5": L5
    }

def stability_score(eigs):

    eigs = np.asarray(eigs, dtype=complex)

    return float(
        np.max(np.real(eigs))
    )



def stability_indicator_at_point(x, y, mu):

    J = jacobian_J(
        x,
        y,
        0.0,
        mu
    )

    eigs = np.linalg.eigvals(J)

    return stability_score(eigs)

def plot_stability_map(
        mu,
        L,
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5),
        N=400,
        rmin=5e-3,
        C_overlay=None
):

    xs = np.linspace(
        xlim[0],
        xlim[1],
        N
    )

    ys = np.linspace(
        ylim[0],
        ylim[1],
        N
    )

    X, Y = np.meshgrid(xs, ys)

    S = np.full(
        X.shape,
        np.nan,
        dtype=float
    )

    potential_surface = np.full(
        X.shape,
        np.nan,
        dtype=float
    )

    for j in range(N):

        for i in range(N):

            x = X[j, i]
            y = Y[j, i]

            r1, r2 = distances(
                x,
                y,
                0.0,
                mu
            )

            if r1 < rmin or r2 < rmin:
                continue

            S[j, i] = (
                stability_indicator_at_point(
                    x,
                    y,
                    mu
                )
            )

            if C_overlay is not None:

                potential_surface[j, i] = (
                    2
                    * omega(
                        x,
                        y,
                        0.0,
                        mu
                    )
                )

    eps = 1e-12

    Splot = np.log10(
        np.maximum(S, eps)
    )

    plt.figure(figsize=(8, 7))

    image = plt.imshow(
        Splot,
        extent=(
            xlim[0],
            xlim[1],
            ylim[0],
            ylim[1]
        ),
        origin="lower",
        aspect="equal"
    )

    plt.colorbar(
        image,
        label="log10(max Re(eigenvalue))"
    )

    plt.scatter(
        [-mu],
        [0],
        marker="o"
    )

    plt.text(
        -mu,
        0,
        "  m1",
        va="center"
    )

    plt.scatter(
        [1 - mu],
        [0],
        marker="o"
    )

    plt.text(
        1 - mu,
        0,
        "  m2",
        va="center"
    )

    for name, point in L.items():

        xL, yL, zL = point

        plt.scatter(
            xL,
            yL,
            marker="x"
        )

        plt.text(
            xL,
            yL,
            f" {name}"
        )

    if C_overlay is not None:

        plt.contour(
            X,
            Y,
            potential_surface,
            levels=[C_overlay],
            linewidths=1
        )

    plt.title(
        "Local instability indicator map\n"
        "log10(max Re(eigenvalue))"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()

    plt.show()

def analyze_and_plot(
        mu,
        x_scan=(-1.5, 1.5),
        n_scan=1200
):

    L = lagrange_points(mu)

    names = [
        "L1",
        "L2",
        "L3",
        "L4",
        "L5"
    ]

    scores = []

    eigs_by_name = {}

    print()
    print("=" * 60)
    print("CR3BP LAGRANGE POINT LINEAR STABILITY")
    print("=" * 60)

    print(
        f"\nmu = {mu}\n"
    )

    for name in names:

        x, y, z = L[name]

        J = jacobian_J(
            x,
            y,
            z,
            mu
        )

        eigs = np.linalg.eigvals(J)

        eigs_by_name[name] = eigs

        score = stability_score(eigs)

        if abs(score) < 1e-12:
            score_plot = 0.0
        else:
            score_plot = score

        scores.append(score_plot)

        print(
            f"{name} at "
            f"({x:.10f}, "
            f"{y:.10f}, "
            f"{z:.10f})"
        )

        eigs_sorted = sorted(
            eigs,
            key=lambda w:
            (
                np.real(w),
                np.imag(w)
            )
        )

        for eig in eigs_sorted:

            print(
                "  lambda = "
                f"{np.real(eig): .6e}"
                "  +  "
                f"{np.imag(eig): .6e} i"
            )

        print(
            "  stability score "
            "max(Re(lambda)) = "
            f"{score:.6e}"
        )

        if score > 1e-10:

            print(
                "  classification: "
                "LINEARLY UNSTABLE"
            )

        else:

            print(
                "  classification: "
                "LINEARLY NEUTRAL/STABLE"
            )

        print()

    plt.figure(figsize=(7, 5))

    plt.axhline(
        0,
        linewidth=1
    )

    plt.bar(
        names,
        scores
    )

    plt.ylabel(
        "max Re(eigenvalue)"
    )

    plt.xlabel(
        "Lagrange point"
    )

    plt.title(
        "Linear Stability of CR3BP Lagrange Points"
    )

    plt.tight_layout()

    xs = np.linspace(
        x_scan[0],
        x_scan[1],
        n_scan
    )

    scan_scores = np.full_like(
        xs,
        np.nan,
        dtype=float
    )

    for i, x in enumerate(xs):

        r1 = abs(x + mu)

        r2 = abs(
            x - (1 - mu)
        )

        if r1 < 5e-3 or r2 < 5e-3:
            continue

        J = jacobian_J(
            x,
            0.0,
            0.0,
            mu
        )

        eigs = np.linalg.eigvals(J)

        scan_scores[i] = (
            stability_score(eigs)
        )

    plt.figure(figsize=(9, 5))

    plt.plot(
        xs,
        scan_scores
    )

    plt.axhline(
        0,
        linewidth=1
    )

    ymax = np.nanmax(scan_scores)

    for name in [
        "L1",
        "L2",
        "L3"
    ]:

        xL = L[name][0]

        plt.axvline(
            xL,
            linewidth=1,
            linestyle="--"
        )

        plt.text(
            xL,
            0.8 * ymax,
            name,
            rotation=90,
            va="top"
        )

    plt.xlabel(
        "x  (y = z = 0)"
    )

    plt.ylabel(
        "max Re(eigenvalue)"
    )

    plt.title(
        "Local Linear Instability Along the x-axis"
    )

    plt.tight_layout()
    

    C_L1 = (
        2
        * omega(
            L["L1"][0],
            L["L1"][1],
            L["L1"][2],
            mu
        )
    )

    plot_stability_map(
        mu,
        L=L,
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5),
        N=400,
        rmin=5e-3,
        C_overlay=C_L1
    )

    plt.show()

    return L, eigs_by_name


if __name__ == "__main__":

    analyze_and_plot(mu)