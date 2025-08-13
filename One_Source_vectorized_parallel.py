#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 09:48:56 2022

Author: Anton

Standalone plume-length calculator and plotter using Mathieu-function series.
This module exposes `run(...)` so it can be imported and executed from a
Jupyter notebook while overriding parameters at call time.
"""

import numpy as np
np.set_printoptions(precision=9)
np.seterr(divide='ignore', invalid='ignore', over='ignore')  # don't print warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import mathieu_functions_OG as mf
import timeit
from datetime import timedelta
import multiprocessing as mp

# === Parameters (defaults) ===
r = 1
alpha_l = 2
alpha_t = 0.05
beta = 1/(2*alpha_l)
C0 = 10
Ca = 8
gamma = 3.5

d = np.sqrt((r*np.sqrt(alpha_l/alpha_t))**2 - r**2)
q = (d**2 * beta**2) / 4
n = 9              # Number of terms in mathieu series - 1
M = 100            # Number of control points

m = mf.mathieu(q)

def uv(x, y):
    """
    Convert Cartesian coordinates (x, y) -> elliptic coordinates (eta, psi).

    Parameters
    ----------
    x : float
        Cartesian x-coordinate.
    y : float
        Cartesian y-coordinate.

    Returns
    -------
    (eta, psi) : tuple[float, float]
        Elliptic radial (eta) and angular (psi) coordinates.
    """
    Y = np.sqrt(alpha_l / alpha_t) * y
    B = x**2 + Y**2 - d**2
    discriminant = B**2 + 4 * d**2 * x**2
    discriminant = max(discriminant, 0.0)
    sqrt_disc = np.sqrt(discriminant)

    p = (-B + sqrt_disc) / (2 * d**2)
    qloc = (-B - sqrt_disc) / (2 * d**2)

    p_clipped = np.clip(p, 0, 1)
    psi_0 = np.arcsin(np.sqrt(p_clipped))

    if Y >= 0 and x >= 0:
        psi = psi_0
    elif Y < 0 and x >= 0:
        psi = np.pi - psi_0
    elif Y <= 0 and x < 0:
        psi = np.pi + psi_0
    else:  # Y > 0 and x < 0
        psi = 2 * np.pi - psi_0

    inner = 1 - 2*qloc + 2 * np.sqrt(max(qloc**2 - qloc, 0.0))
    eta = 0.5 * np.log(inner)
    return eta, psi


def Se(order, psi):
    """
    Even angular Mathieu function of the first kind (real part).

    Parameters
    ----------
    order : int
        Series order n ≥ 0.
    psi : float
        Elliptic angular coordinate.

    Returns
    -------
    float
        ce_n(psi; q) evaluated using the global `m`.
    """
    return m.ce(order, psi).real


def So(order, psi):
    """
    Odd angular Mathieu function of the first kind (real part).

    Parameters
    ----------
    order : int
        Series order n ≥ 1.
    psi : float
        Elliptic angular coordinate.

    Returns
    -------
    float
        se_n(psi; q) evaluated using the global `m`.
    """
    return m.se(order, psi).real


def Ye(order, eta):
    """
    Even radial Mathieu function of the second kind (real part).

    Parameters
    ----------
    order : int
        Series order n ≥ 0.
    eta : float
        Elliptic radial coordinate.

    Returns
    -------
    float
        Ke_n(eta; q) evaluated using the global `m`.
    """
    return m.Ke(order, eta).real


def Yo(order, eta):
    """
    Odd radial Mathieu function of the second kind (real part).

    Parameters
    ----------
    order : int
        Series order n ≥ 1.
    eta : float
        Elliptic radial coordinate.

    Returns
    -------
    float
        Ko_n(eta; q) evaluated using the global `m`.
    """
    return m.Ko(order, eta).real


# will be recomputed in run()
Coeff = None

def F1(x1):
    """
    Boundary target function along the element (used to fit coefficients).

    Parameters
    ----------
    x1 : float
        Local x-coordinate on the circular boundary.

    Returns
    -------
    float
        Target concentration value at the boundary point.
    """
    return (C0*gamma + Ca) * np.exp(-beta * x1)


def c(x, y):
    """
    Evaluate the concentration field at a point.

    Parameters
    ----------
    x : float
        Cartesian x-coordinate.
    y : float
        Cartesian y-coordinate.

    Returns
    -------
    float
        Concentration at (x, y).
    """
    if (x**2 + y**2) <= r**2:
        return C0

    eta, psi = uv(x, y)

    F = Coeff[0][0]*Se(0, psi)*Ye(0, eta)
    for w in range(1, n):
        F += Coeff[0][2*w-1]*So(w, psi)*Yo(w, eta) \
           + Coeff[0][2*w]*Se(w, psi)*Ye(w, eta)

    val = F * np.exp(beta*x)
    if val > Ca:
        return (((val) - Ca) / gamma).round(9)
    else:
        return (val - Ca).round(9)


def compute_conc(point):
    """
    Thin wrapper for multiprocessing to compute concentration at a point.

    Parameters
    ----------
    point : tuple[float, float]
        (x, y) coordinate pair.

    Returns
    -------
    float
        Concentration c(x, y).
    """
    x, y = point
    return c(x, y)

def _init_mp_worker(r_param, alpha_l_param, alpha_t_param, C0_param, Ca_param, gamma_param,
                    n_param, M_param, beta_param, d_param, q_param, coeff_param):
    """
    Initialize module-level globals inside each multiprocessing worker so that
    uv(), Se/So/Ye/Yo(), and c() see the same state as the parent process.
    """
    global r, alpha_l, alpha_t, C0, Ca, gamma, n, M, beta, d, q, m, Coeff
    r = float(r_param)
    alpha_l = float(alpha_l_param)
    alpha_t = float(alpha_t_param)
    C0 = float(C0_param)
    Ca = float(Ca_param)
    gamma = float(gamma_param)
    n = int(n_param)
    M = int(M_param)
    beta = float(beta_param)
    d = float(d_param)
    q = float(q_param)
    Coeff = coeff_param
    m = mf.mathieu(q)

def Conc_array(x_min, x_max, y_min, y_max, inc):
    """
     Compute the concentration field on a rectangular grid using multiprocessing.

     Parameters
     ----------
     x_min, x_max : float
         Grid domain in x; samples are taken in [x_min, x_max) with step `inc`.
     y_min, y_max : float
         Grid domain in y; samples are taken in [y_min, y_max) with step `inc`.
     inc : float
         Grid spacing in both directions.

     Returns
     -------
     (xaxis, yaxis, Conc) : tuple[np.ndarray, np.ndarray, np.ndarray]
         xaxis : shape (Nx,)
         yaxis : shape (Ny,)
         Conc  : shape (Ny, Nx) array of concentrations.
     """
    xaxis = np.arange(x_min, x_max, inc, dtype=float)
    yaxis = np.arange(y_min, y_max, inc, dtype=float)
    X, Y = np.meshgrid(xaxis, yaxis)

    points = list(zip(X.ravel(), Y.ravel()))

    with mp.Pool(
        processes=mp.cpu_count(),
        initializer=_init_mp_worker,
        initargs=(r, alpha_l, alpha_t, C0, Ca, gamma, n, M, beta, d, q, Coeff)
    ) as pool:
        Conc_flat = pool.map(compute_conc, points)

    # <- force float, then reshape by axis lengths (Ny, Nx)
    Conc = np.asarray(Conc_flat, dtype=float).reshape(len(yaxis), len(xaxis))

    return xaxis, yaxis, Conc

def run(
    *,
    r_param=r,
    alpha_l_param=alpha_l,
    alpha_t_param=alpha_t,
    C0_param=C0,
    Ca_param=Ca,
    gamma_param=gamma,
    n_param=n,
    M_param=M,
    x_min=0, x_max=400, y_min=-5, y_max=5, inc=0.5
):
    """
    Execute the end-to-end plume computation and plotting.

    Parameters
    ----------
    r_param : float, default=r
        Source radius.
    alpha_l_param : float, default=alpha_l
        Longitudinal dispersivity.
    alpha_t_param : float, default=alpha_t
        Transverse dispersivity.
    C0_param : float, default=C0
        Source concentration.
    Ca_param : float, default=Ca
        Background acceptor concentration magnitude.
    gamma_param : float, default=gamma
        Stoichiometric factor.
    n_param : int, default=n
        Number of series orders minus one (i.e., orders 0..n-1).
    M_param : int, default=M
        Number of boundary control points.
    x_min, x_max, y_min, y_max : float
        Rectangular domain limits for the field evaluation.
    inc : float
        Grid spacing for the field evaluation.

    Returns
    -------
    None
        Displays figures and prints summary;
    """
    global r, alpha_l, alpha_t, beta, C0, Ca, gamma, d, q, n, M, m, Coeff

    # Override parameters
    r       = float(r_param)
    alpha_l = float(alpha_l_param)
    alpha_t = float(alpha_t_param)
    C0      = float(C0_param)
    Ca      = float(Ca_param)
    gamma   = float(gamma_param)
    n       = int(n_param)
    M       = int(M_param)

    # Recompute derived quantities and refresh Mathieu object
    beta = 1.0 / (2.0 * alpha_l)
    d = np.sqrt((r*np.sqrt(alpha_l/alpha_t))**2 - r**2)
    q = (d**2 * beta**2) / 4.0
    m = mf.mathieu(q)

    # Rebuild coefficients with current parameters
    phi = np.linspace(0, 2*np.pi, M)
    x1 = r*np.cos(phi)
    y1 = r*np.sin(phi)

    uv_vec = np.vectorize(uv)
    eta1, psi1 = uv_vec(x1, y1)

    lst = []
    for i in range(0, M):
        lst.append(Se(0, psi1[i]) * Ye(0, eta1[i]))
        for j in range(1, n):
            lst.append(So(j, psi1[i]) * Yo(j, eta1[i]))
            lst.append(Se(j, psi1[i]) * Ye(j, eta1[i]))

    F_M = []
    s = 2*n - 1
    for k in range(0, len(lst), s):
        F_M.append(lst[k:k+s])

    F = [F1(x1[u]) for u in range(0, M)]
    Coeff = np.linalg.lstsq(F_M, F, rcond=None)

    # Compute grid and plot
    start = timeit.default_timer()
    result = Conc_array(x_min, x_max + inc, y_min, y_max + inc, inc)
    stop = timeit.default_timer()

    sec = int(stop - start)
    cpu_time = timedelta(seconds=sec)
    print('Computation time [hh:mm:ss]:', cpu_time)

    plt.figure(figsize=(16, 9), dpi=300, layout="constrained")
    mpl.rcParams.update({'font.size': 22})
    plt.xlabel('$x$ (m)')
    plt.ylabel('$y$ (m)')
    plt.xticks(range(len(result[0]))[::int(50/inc)], result[0][::int(50/inc)].round(0))
    plt.yticks(range(len(result[1]))[::int(10/inc)], result[1][::int(10/inc)].round(0))

    Plume_cd = plt.contourf(result[2], levels=np.linspace(0, C0+0.01, 11), cmap='Reds')
    Plume_ca = plt.contourf(result[2], levels=np.linspace(-Ca, 0, 9), cmap='Blues_r')
    Plume_max = plt.contour(result[2], levels=[0], linewidths=2, colors='k')

    cbar_cd = plt.colorbar(Plume_cd, ticks=Plume_cd.levels, label='Electron donor concentration [mg/l]', location='bottom', aspect=75)
    cbar_ca = plt.colorbar(Plume_ca, ticks=Plume_ca.levels, label='Electron acceptor concentration [mg/l]', location='bottom', aspect=75)
    cbar_ca.set_ticks(Plume_ca.levels)
    cbar_ca.set_ticklabels([f"{abs(level):.0f}" for level in Plume_ca.levels])

    bar_height = 0.01
    bar_width = 0.8
    bar_x = 0.1
    cbar_ca.ax.set_position([bar_x, 0, bar_width, bar_height])
    cbar_cd.ax.set_position([bar_x, 1, bar_width, bar_height])

    Lmax = Plume_max.get_paths()[0]
    print('Lmax =', int(np.max(Lmax.vertices[:, int((result[1][0]+result[1][-1])/2)])*inc - np.abs(result[0][0])))

    plt.show()

    # Boundary stats
    phi2 = np.linspace(0, 2*np.pi, 360)
    x_test = (r + 1e-9) * np.cos(phi2)
    y_test = (r + 1e-9) * np.sin(phi2)

    Err = [c(x_test[i], y_test[i]) for i in range(0, 360, 1)]
    print('Min =', np.min(Err).round(9), 'mg/l')
    print('Max =', np.max(Err).round(9), 'mg/l')
    print('Mean =', np.mean(Err).round(9), 'mg/l')
    print('Standard Deviation =', np.std(Err).round(15), 'mg/l')

    plt.figure(figsize=(16,9), dpi=300)
    mpl.rcParams.update({'font.size': 22})
    plt.plot(phi2, Err, color='k')
    plt.xlabel('Angle (°)')
    plt.ylabel('Concentration (mg/l)')
    plt.ticklabel_format(axis='both', style='scientific', useMathText=True, useOffset=True, scilimits=(0,2))
    plt.xticks(np.linspace(0, 2*np.pi, 7), np.linspace(0, 360, 7))
    plt.xlim([0, 2*np.pi])
    # No return value by design.

if __name__ == '__main__':
    run()
