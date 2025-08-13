#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 09:48:56 2022

@author: anton
"""
import numpy as np
np.seterr(divide='ignore', invalid='ignore', over='ignore', under='ignore')                #dont print warnings
np.set_printoptions(precision=9)
import matplotlib as mpl
import matplotlib.pyplot as plt
import mathieu_functions_OG as mf
import timeit
from datetime import timedelta
import multiprocessing as mp

#Parameter
alpha_l = 2
alpha_t = 0.05
beta = 1/(2*alpha_l)
C0 = 10
C1 = -8               #-8 for acceptor source
Ca = 8
gamma = 3.5
r = 1

d = np.sqrt((r*np.sqrt(alpha_l/alpha_t))**2-r**2)
q = (d**2*beta**2)/4

n = 7            #Number of terms in mathieu series
M = 100           #Number of Control Points, 5x overspecification

#Mathieu Functions
m = mf.mathieu(q)

#Real Mathieu Functions
def Se(order, psi):                    #even angular first kind
    return m.ce(order, psi).real
def So(order, psi):                    #odd angular first kind
    return m.se(order, psi).real
def Ye(order, eta):                    #even radial second Kind
    return m.Ke(order, eta).real
def Yo(order, eta):                    #odd radial second Kind
    return m.Ko(order, eta).real

#wrapper xy to eta psi
def uv(x, y):
    """
    Convert Cartesian (x, y) to elliptic coordinates (eta, psi),
    with foci at (±d, 0) and optional anisotropy.

    Parameters:
        x, y     : Cartesian coordinates
        d        : half-distance between foci

    Returns:
        eta, psi : elliptic coordinates
    """
    Y = np.sqrt(alpha_l / alpha_t) * y
    B = x**2 + Y**2 - d**2
    discriminant = B**2 + 4 * d**2 * x**2

    # Safety against numerical issues
    discriminant = max(discriminant, 0.0)
    sqrt_disc = np.sqrt(discriminant)

    p = (-B + sqrt_disc) / (2 * d**2)
    q = (-B - sqrt_disc) / (2 * d**2)

    # Clip p to [0, 1] for arcsin safety
    p_clipped = np.clip(p, 0, 1)
    psi_0 = np.arcsin(np.sqrt(p_clipped))

    # Determine correct quadrant
    if Y >= 0 and x >= 0:
        psi = psi_0
    elif Y < 0 and x >= 0:
        psi = np.pi - psi_0
    elif Y <= 0 and x < 0:
        psi = np.pi + psi_0
    else:  # Y > 0 and x < 0
        psi = 2 * np.pi - psi_0

    # Compute eta
    inner = 1 - 2*q + 2 * np.sqrt(max(q**2 - q, 0.0))
    eta = 0.5 * np.log(inner)

    return eta, psi

#polar coordinates
phi = np.linspace(0, 2*np.pi, M)
x1 = r*np.cos(phi)
y1 = r*np.sin(phi)

#source coordinates xy and distance of second source
D1 = 75
D2 = 0
x2 = x1 - D1
y2 = y1 - D2
x3 = x1 + D1
y3 = y1 + D2

#source coordinates eta psi
uv_vec = np.vectorize(uv)
psi1 = uv_vec(x1, y1)[1]
psi2 = uv_vec(x2, y2)[1]
psi3 = uv_vec(x3, y3)[1]
eta1 = uv_vec(x1, y1)[0]
eta2 = uv_vec(x2, y2)[0]
eta3 = uv_vec(x3, y3)[0]

#%%
#general target function:
def F_target(x, Ci):
    if Ci > 0:
        return (Ci*gamma+Ca)*np.exp(-beta*x), (Ci), 'r'
    if Ci <= 0:
        return (Ci+Ca)*np.exp(-beta*x), (Ci), 'b'

#System of Equations to calculate coefficients
#"perspective" source 1
lst = []                                #empty array

for i in range(0, M):                                    #filling array with all terms of MF for 1st source
    for j in range(0, 1):
        lst.append(Se(j, psi1[i])*Ye(j, eta1[i]))
    for j in range(1, n):
        lst.append(So(j, psi1[i])*Yo(j, eta1[i]))
        lst.append(Se(j, psi1[i])*Ye(j, eta1[i]))
    for j in range(0, 1):                                #filling array with all terms of MF for 2nd source
        lst.append(Se(j, psi2[i])*Ye(j, eta2[i]))
    for j in range(1, n):
        lst.append(So(j, psi2[i])*Yo(j, eta2[i]))
        lst.append(Se(j, psi2[i])*Ye(j, eta2[i]))

F_M1 = []
s = (2*n-1)*2
for k in range(0, len(lst), s):           #appending each line (s elements) as arrays (in brackets) -> achieve right array structure (list of arrays)
    F_M1.append(lst[k:k+s])

#"perspective" source 2
lst2 = []

for i in range(0, M):
    for j in range(0, 1):                                   #filling array with all terms of MF for 1st source
        lst2.append(Se(j, psi3[i])*Ye(j, eta3[i]))
    for j in range(1, n):
        lst2.append(So(j, psi3[i])*Yo(j, eta3[i]))
        lst2.append(Se(j, psi3[i])*Ye(j, eta3[i]))
    for j in range(0, 1):                                   #filling array with all terms of MF for 2nd source
        lst2.append(Se(j, psi1[i])*Ye(j, eta1[i]))
    for j in range(1, n):
        lst2.append(So(j, psi1[i])*Yo(j, eta1[i]))
        lst2.append(Se(j, psi1[i])*Ye(j, eta1[i]))

#%%
F_M2 = []
s = (2*n-1)*2
for k in range(0, len(lst2), s):          #appending each line (s elements) as lists (in brackets) -> achieve right array structure (list of arrays)
    F_M2.append(lst2[k:k+s])

F_M = F_M1 + F_M2                       #combining arrays for "perspective"" 1 and 2

F = []                                  #target function vector

for u in range(0, M):
    F.append(F_target(x1[u], C0)[0])
for v in range(0, M):
    F.append(F_target(x3[v], C1)[0])

Coeff = np.linalg.lstsq(F_M, F, rcond=None)
# print(Coeff[0])

#%%
def c(x, y):
    if (x**2+y**2)<=r**2:
        return F_target(x1[u], C0)[1]
    if ((x-D1)**2+(y-D2)**2)<=r**2:
        return F_target(x3[v], C1)[1]

    psi = uv(x, y)[1]
    eta = uv(x, y)[0]
    psi2 = uv(x-D1, y-D2)[1]
    eta2 = uv(x-D1, y-D2)[0]

    F1 = Coeff[0][0]*Se(0, psi)*Ye(0, eta)
    for w in range(1, n):
        F1 += Coeff[0][2*w-1]*So(w, psi)*Yo(w ,eta) \
            + Coeff[0][2*w]*Se(w, psi)*Ye(w, eta)

    F2 = Coeff[0][2*n-1]*Se(0, psi2)*Ye(0, eta2)
    for b in range(1, n):
        F2 += Coeff[0][(2*n-1)+(2*b-1)]*So(b, psi2)*Yo(b, eta2) \
            + Coeff[0][(2*n-1)+(2*b)]*Se(b, psi2)*Ye(b, eta2)                   #till here F domain

    # return (F1*np.exp(beta*x) + F2*np.exp(beta*x)).round(9)              #from here C domain

    if (F1*np.exp(beta*x) + F2*np.exp(beta*x))> Ca:
        return (((F1*np.exp(beta*x) + F2*np.exp(beta*x))-Ca)/gamma).round(9)
    else:
        return ((F1*np.exp(beta*x) + F2*np.exp(beta*x))-Ca).round(9)
#%%
# #concentration array for plotting purpose

inc = 0.1
# Define a helper function for `Pool.map`
def compute_conc(point):
    x, y = point
    return c(x, y)

def Conc_array(x_min, x_max, y_min, y_max, inc):
    xaxis = np.arange(x_min, x_max, inc)
    yaxis = np.arange(y_min, y_max, inc)
    X, Y = np.meshgrid(xaxis, yaxis)

    # Flatten the grid for parallel processing
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    # Prepare the inputs as tuples of (x, y) for the function `c`
    points = list(zip(X_flat, Y_flat))

    # Use multiprocessing.Pool to parallelize the computation of `c(x, y)` over all grid points
    with mp.Pool(mp.cpu_count()) as pool:
        Conc_flat = pool.map(compute_conc, points)

    # Reshape the flat result back to the original grid shape
    Conc = np.array(Conc_flat).reshape(X.shape)

    return xaxis, yaxis, Conc

# Run the function
def run():
    start = timeit.default_timer()

    result = Conc_array(0, 100+inc, -5, 5+inc, inc)

    stop = timeit.default_timer()
    sec = int(stop - start)
    cpu_time = timedelta(seconds = sec)
    print('Computation time [hh:mm:ss]:', cpu_time)
#%% plotting

    plt.figure(figsize=(16, 9), dpi = 300, layout='constrained')
    mpl.rcParams.update({'font.size': 22})
    # plt.axis('scaled')
    plt.xlabel('$x$ (m)')
    plt.ylabel('$y$ (m)')

    plt.xticks(range(len(result[0]))[::int(50/inc)], result[0][::int(50/inc)].round(0))
    plt.yticks(range(len(result[1]))[::int(10/inc)], result[1][::int(10/inc)].round(0))
    Plume_cd = plt.contourf(result[2], levels=np.linspace(0, C0, 11), cmap='Reds')
    Plume_ca = plt.contourf(result[2], levels=np.linspace(-8.01, 0, 9), cmap='Blues_r')
    Plume_max = plt.contour(result[2], levels=[0], linewidths=2, colors='k')

    #Colorbar
    cbar_cd = plt.colorbar(Plume_cd, ticks=Plume_cd.levels, label='Electron donor concentration [mg/l]', location='bottom', aspect=75)
    cbar_ca = plt.colorbar(Plume_ca, ticks=Plume_ca.levels, label='Electron acceptor concentration [mg/l]', location='bottom', aspect=75)
    cbar_ca.set_ticks(Plume_ca.levels)  # Ensure it uses the same tick positions
    cbar_ca.set_ticklabels([f"{abs(level):.0f}" for level in Plume_ca.levels])

    # plt.subplots_adjust(bottom=0)  # Increase if needed

    # Get one of the original colorbar positions to reuse width/height
    bar_height = 0.01
    bar_width = 0.8
    bar_x = 0.1

    # Set tighter vertical positions
    cbar_ca.ax.set_position([bar_x, 0, bar_width, bar_height])  # Acceptor (top one)
    cbar_cd.ax.set_position([bar_x, 1, bar_width, bar_height])  # Donor (bottom one)


    Lmax = Plume_max.get_paths()[0]
    print('Lmax =', int(np.max(Lmax.vertices[:,:])*inc))
    plt.show()

#%%
#absolut error [mg/l]
    phi2 = np.linspace(0, 2*np.pi, 360)
    x_test = (r + 1e-9) * np.cos(phi2)
    y_test = (r + 1e-9) * np.sin(phi2)

    x2_test = x_test+D1
    y2_test = y_test+D2

    Err = []
    Err2 = []
    for i in range(0,360,1):
        Err.append((c(x_test[i], y_test[i])))
        Err2.append((c(x2_test[i], y2_test[i])))
    #print(Err)
    #print(Err2)
    print('Min =',np.min(Err).round(9), 'mg/l')
    print('Max =',np.max(Err).round(9), 'mg/l')
    print('Mean =',np.mean(Err).round(9), 'mg/l')
    print('Standard Deviation =',np.std(Err).round(9), 'mg/l')
    print('Min2 =',np.min(Err2).round(9), 'mg/l')
    print('Max2 =',np.max(Err2).round(9), 'mg/l')
    print('Mean2 =',np.mean(Err2).round(9), 'mg/l')
    print('Standard Deviation2 =',np.std(Err2).round(9), 'mg/l')

    plt.figure(figsize=(16,9), dpi=300)
    mpl.rcParams.update({'font.size': 22})
    plt.plot(phi2,Err, color='black', linewidth=2, label='element 1')
    plt.plot(phi2,Err2, color='black', linewidth=2, linestyle = '--', label='element 2')
    plt.xlim([0, 2*np.pi])
    plt.xlabel('Angle (°)')
    plt.ylabel('Concentration (mg/l)')
    plt.xticks(np.linspace(0, 2*np.pi, 13), np.linspace(0, 360, 13).astype(int))
    plt.legend()
    return result

if __name__ ==  '__main__':
    run()