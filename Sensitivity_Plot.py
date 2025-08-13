# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:07:12 2025

@author: Anton
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})

# Data
C0_mu_star = [3655, 3680, 3492, 3458, 3460]
C0_sigma = [4833, 4804, 4672, 4677, 4714]

alpha_t_mu_star = [1356, 1379, 1296, 1477, 1270]
alpha_t_sigma = [2621, 2599, 2486, 2811, 2504]

gamma_mu_star = [3402, 3855, 3629, 3746, 3505]
gamma_sigma = [4569, 4866, 4756, 4737, 4696]

r_mu_star = [3018, 2882, 3184, 3092, 3213]
r_sigma = [4813, 4694, 4832, 4803, 4945]

# Plot
plt.figure(figsize=(16, 9), dpi=300)

# Plotting each parameter
plt.scatter(C0_mu_star, C0_sigma, marker='o', facecolors='None', edgecolors='r', s=200, linewidth=2, label='$C_0$')
plt.scatter(alpha_t_mu_star, alpha_t_sigma, marker='s', facecolors='None', edgecolors='g', s=200, linewidth=2, label='$\\alpha_T$')
plt.scatter(gamma_mu_star, gamma_sigma, marker='^', facecolors='None', edgecolors='b', s=200, linewidth=2, label='$\\gamma$')
plt.scatter(r_mu_star, r_sigma, marker='d', facecolors='None', edgecolors='orange', s=200, linewidth=2, label='$R$')

# Adding labels and legend
plt.xlim((1000, 5000))
plt.ylim((1000, 5000))
plt.xlabel('$\\mu^\\star$')
plt.ylabel('$\\sigma$')
plt.axline((0,0), slope=1, color = 'k', linewidth=0.75, linestyle = '--')
plt.fill_between(np.linspace(1000,5000,100), np.linspace(1000,5000,100), 1000, color='none', hatch='\\',edgecolor='grey', alpha=0.3)
plt.legend(fontsize=24)
#plt.grid(True)

# Show plot
plt.tight_layout()
plt.savefig('fig6.pdf')
plt.show()
