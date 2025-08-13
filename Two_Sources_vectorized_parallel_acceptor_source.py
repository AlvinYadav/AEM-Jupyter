#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Donor (+C0) and acceptor (C1 < 0) two-source plume simulation, Mathieu-series.
"""

import numpy as np
np.seterr(divide='ignore', invalid='ignore', over='ignore', under='ignore')
np.set_printoptions(precision=9)
import matplotlib as mpl
import matplotlib.pyplot as plt
import mathieu_functions_OG as mf
import timeit
from datetime import timedelta
import multiprocessing as mp

# --- Defaults (can be overridden via run(...)) ---
alpha_l = 2.0
alpha_t = 0.05
beta = 1.0 / (2.0 * alpha_l)
C0 = 10.0
C1 = -8.0
Ca = 8.0
gamma = 3.5
r = 1.0

d = np.sqrt((r*np.sqrt(alpha_l/alpha_t))**2 - r**2)
q = (d**2 * beta**2) / 4.0

n = 7
M = 100

D1 = 75.0
D2 = 0.0

m = mf.mathieu(q)
Coeff = None


def Se(order, psi):
    """Even angular Mathieu, real part."""
    return m.ce(order, psi).real


def So(order, psi):
    """Odd angular Mathieu, real part."""
    return m.se(order, psi).real


def Ye(order, eta):
    """Even radial Mathieu of 2nd kind, real part."""
    return m.Ke(order, eta).real


def Yo(order, eta):
    """Odd radial Mathieu of 2nd kind, real part."""
    return m.Ko(order, eta).real


def uv(x, y):
    """
    Cartesian (x,y) -> elliptic (eta, psi) using global anisotropy and foci.
    """
    Y = np.sqrt(alpha_l / alpha_t) * y
    B = x**2 + Y**2 - d**2
    disc = max(B**2 + 4 * d**2 * x**2, 0.0)
    sqrt_disc = np.sqrt(disc)

    p = (-B + sqrt_disc) / (2 * d**2)
    qloc = (-B - sqrt_disc) / (2 * d**2)

    psi_0 = np.arcsin(np.sqrt(np.clip(p, 0.0, 1.0)))
    if Y >= 0 and x >= 0:
        psi = psi_0
    elif Y < 0 and x >= 0:
        psi = np.pi - psi_0
    elif Y <= 0 and x < 0:
        psi = np.pi + psi_0
    else:
        psi = 2 * np.pi - psi_0

    inner = 1 - 2*qloc + 2 * np.sqrt(max(qloc**2 - qloc, 0.0))
    eta = 0.5 * np.log(inner)
    return eta, psi


def F_target(x, Ci):
    """
    Boundary target function and metadata for source strength Ci.
    """
    if Ci > 0:
        return (Ci*gamma + Ca) * np.exp(-beta * x), Ci, 'r'
    else:
        return (Ci + Ca) * np.exp(-beta * x), Ci, 'b'


def c(x, y):
    """
    Concentration at (x,y) for donor at (0,0) and acceptor at (D1,D2).
    """
    if (x**2 + y**2) <= r**2:
        return C0
    if ((x - D1)**2 + (y - D2)**2) <= r**2:
        return C1

    eta,  psi  = uv(x, y)
    eta2, psi2 = uv(x - D1, y - D2)

    F1 = Coeff[0][0]*Se(0, psi)*Ye(0, eta)
    for w in range(1, n):
        F1 += Coeff[0][2*w-1]*So(w, psi)*Yo(w, eta) + Coeff[0][2*w]*Se(w, psi)*Ye(w, eta)

    F2 = Coeff[0][2*n-1]*Se(0, psi2)*Ye(0, eta2)
    for b in range(1, n):
        base = (2*n-1)
        F2 += Coeff[0][base+(2*b-1)]*So(b, psi2)*Yo(b, eta2) + Coeff[0][base+(2*b)]*Se(b, psi2)*Ye(b, eta2)

    val = F1*np.exp(beta*x) + F2*np.exp(beta*x)
    return (((val - Ca)/gamma) if val > Ca else (val - Ca)).round(9)


def compute_conc(point):
    """Wrapper for multiprocessing map()."""
    x, y = point
    return c(x, y)


def _init_mp_worker(r_param, alpha_l_param, alpha_t_param, beta_param, C0_param, C1_param,
                    Ca_param, gamma_param, n_param, M_param, d_param, q_param, D1_param,
                    D2_param, coeff_param):
    """
    Initialize globals inside each worker process.
    """
    global r, alpha_l, alpha_t, beta, C0, C1, Ca, gamma, n, M, d, q, D1, D2, m, Coeff
    r = float(r_param)
    alpha_l = float(alpha_l_param)
    alpha_t = float(alpha_t_param)
    beta = float(beta_param)
    C0 = float(C0_param)
    C1 = float(C1_param)
    Ca = float(Ca_param)
    gamma = float(gamma_param)
    n = int(n_param)
    M = int(M_param)
    d = float(d_param)
    q = float(q_param)
    D1 = float(D1_param)
    D2 = float(D2_param)
    Coeff = coeff_param
    m = mf.mathieu(q)


def Conc_array(x_min, x_max, y_min, y_max, inc):
    """
    Compute (xaxis, yaxis, Z) for the domain with spacing inc.
    """
    xaxis = np.arange(x_min, x_max, inc, dtype=float)
    yaxis = np.arange(y_min, y_max, inc, dtype=float)
    X, Y = np.meshgrid(xaxis, yaxis)
    points = list(zip(X.ravel(), Y.ravel()))

    with mp.Pool(
        processes=mp.cpu_count(),
        initializer=_init_mp_worker,
        initargs=(r, alpha_l, alpha_t, beta, C0, C1, Ca, gamma, n, M, d, q, D1, D2, Coeff)
    ) as pool:
        Conc_flat = pool.map(compute_conc, points)

    Z = np.asarray(Conc_flat, dtype=float).reshape(len(yaxis), len(xaxis))
    return xaxis, yaxis, Z


def run(
    *,
    r_param=r,
    alpha_l_param=alpha_l,
    alpha_t_param=alpha_t,
    C0_param=C0,
    C1_param=C1,
    Ca_param=Ca,
    gamma_param=gamma,
    n_param=n,
    M_param=M,
    D1_param=D1,
    D2_param=D2,
    x_min=0.0, x_max=100.0, y_min=-5.0, y_max=5.0, inc=0.1
):
    """
    Execute donor (+C0) and acceptor (C1<0) two-source simulation and plots.
    """
    global r, alpha_l, alpha_t, beta, C0, C1, Ca, gamma, n, M, d, q, D1, D2, m, Coeff

    r       = float(r_param)
    alpha_l = float(alpha_l_param)
    alpha_t = float(alpha_t_param)
    C0      = float(C0_param)
    C1      = float(C1_param)
    Ca      = float(Ca_param)
    gamma   = float(gamma_param)
    n       = int(n_param)
    M       = int(M_param)
    D1      = float(D1_param)
    D2      = float(D2_param)

    beta = 1.0 / (2.0 * alpha_l)
    d = np.sqrt((r*np.sqrt(alpha_l/alpha_t))**2 - r**2)
    q = (d**2 * beta**2) / 4.0
    m = mf.mathieu(q)

    phi = np.linspace(0, 2*np.pi, M)
    x1 = r*np.cos(phi); y1 = r*np.sin(phi)
    x2 = x1 - D1; y2 = y1 - D2
    x3 = x1 + D1; y3 = y1 + D2

    uv_vec = np.vectorize(uv)
    eta1, psi1 = uv_vec(x1, y1)
    eta2v, psi2v = uv_vec(x2, y2)
    eta3v, psi3v = uv_vec(x3, y3)

    lst = []
    for i in range(M):
        lst.append(Se(0, psi1[i])*Ye(0, eta1[i]))
        for j in range(1, n):
            lst.append(So(j, psi1[i])*Yo(j, eta1[i]))
            lst.append(Se(j, psi1[i])*Ye(j, eta1[i]))
        lst.append(Se(0, psi2v[i])*Ye(0, eta2v[i]))
        for j in range(1, n):
            lst.append(So(j, psi2v[i])*Yo(j, eta2v[i]))
            lst.append(Se(j, psi2v[i])*Ye(j, eta2v[i]))
    F_M1 = []
    s = (2*n - 1) * 2
    for k in range(0, len(lst), s):
        F_M1.append(lst[k:k+s])

    lst2 = []
    for i in range(M):
        lst2.append(Se(0, psi3v[i])*Ye(0, eta3v[i]))
        for j in range(1, n):
            lst2.append(So(j, psi3v[i])*Yo(j, eta3v[i]))
            lst2.append(Se(j, psi3v[i])*Ye(j, eta3v[i]))
        lst2.append(Se(0, psi1[i])*Ye(0, eta1[i]))
        for j in range(1, n):
            lst2.append(So(j, psi1[i])*Yo(j, eta1[i]))
            lst2.append(Se(j, psi1[i])*Ye(j, eta1[i]))
    F_M2 = []
    for k in range(0, len(lst2), s):
        F_M2.append(lst2[k:k+s])

    F_M = F_M1 + F_M2

    F = []
    for u in range(M):
        F.append(F_target(x1[u], C0)[0])
    for v in range(M):
        F.append(F_target(x3[v], C1)[0])

    Coeff = np.linalg.lstsq(F_M, F, rcond=None)

    start = timeit.default_timer()
    xaxis, yaxis, Z = Conc_array(x_min, x_max + inc, y_min, y_max + inc, inc)
    elapsed = int(timeit.default_timer() - start)
    print("Computation time [hh:mm:ss]:", timedelta(seconds=elapsed))

    Xg, Yg = np.meshgrid(xaxis, yaxis)
    plt.figure(figsize=(16, 9), dpi=300, layout='constrained')
    mpl.rcParams.update({'font.size': 22})
    plt.xlabel('$x$ (m)'); plt.ylabel('$y$ (m)')
    plt.xticks(range(len(xaxis))[::max(1, int(50/inc))], xaxis[::max(1, int(50/inc))].round(0))
    plt.yticks(range(len(yaxis))[::max(1, int(10/inc))],  yaxis[::max(1, int(10/inc))].round(0))

    Plume_cd  = plt.contourf(Xg, Yg, Z, levels=np.linspace(0, C0, 11), cmap='Reds')
    Plume_ca  = plt.contourf(Xg, Yg, Z, levels=np.linspace(-8.01, 0, 9), cmap='Blues_r')
    Plume_max = plt.contour( Xg, Yg, Z, levels=[0], linewidths=2, colors='k')

    cbar_cd = plt.colorbar(Plume_cd, ticks=Plume_cd.levels, label='Electron donor concentration [mg/l]', location='bottom', aspect=75)
    cbar_ca = plt.colorbar(Plume_ca, ticks=Plume_ca.levels, label='Electron acceptor concentration [mg/l]', location='bottom', aspect=75)
    cbar_ca.set_ticks(Plume_ca.levels)
    cbar_ca.set_ticklabels([f"{abs(level):.0f}" for level in Plume_ca.levels])

    bar_height, bar_width, bar_x = 0.01, 0.8, 0.1
    cbar_ca.ax.set_position([bar_x, 0, bar_width, bar_height])
    cbar_cd.ax.set_position([bar_x, 1, bar_width, bar_height])

    Lmax = Plume_max.get_paths()[0]
    print('Lmax =', int(np.max(Lmax.vertices[:,:]) * inc))
    plt.show()

    # error checks
    phi2 = np.linspace(0, 2*np.pi, 360)
    x_test = (r + 1e-9) * np.cos(phi2)
    y_test = (r + 1e-9) * np.sin(phi2)
    x2_test = x_test + D1
    y2_test = y_test + D2

    Err  = [c(x_test[i],  y_test[i])  for i in range(0, 360)]
    Err2 = [c(x2_test[i], y2_test[i]) for i in range(0, 360)]
    print('Min =', np.min(Err).round(9),  'mg/l')
    print('Max =', np.max(Err).round(9),  'mg/l')
    print('Mean =', np.mean(Err).round(9), 'mg/l')
    print('Standard Deviation =', np.std(Err).round(9), 'mg/l')
    print('Min2 =', np.min(Err2).round(9), 'mg/l')
    print('Max2 =', np.max(Err2).round(9), 'mg/l')
    print('Mean2 =', np.mean(Err2).round(9), 'mg/l')
    print('Standard Deviation2 =', np.std(Err2).round(9), 'mg/l')

    plt.figure(figsize=(16,9), dpi=300)
    mpl.rcParams.update({'font.size': 22})
    plt.plot(phi2, Err,  color='black', linewidth=2, label='element 1')
    plt.plot(phi2, Err2, color='black', linewidth=2, linestyle='--', label='element 2')
    plt.xlim([0, 2*np.pi])
    plt.xlabel('Angle (°)'); plt.ylabel('Concentration (mg/l)')
    plt.xticks(np.linspace(0, 2*np.pi, 13), np.linspace(0, 360, 13).astype(int))
    plt.legend()
    # no return


if __name__ == '__main__':
    run()
