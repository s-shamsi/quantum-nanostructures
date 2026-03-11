#!/usr/bin/env python
# coding: utf-8
"""
circular_annulus.py
===================
Computes and plots the probability density functions for a quantum particle
confined in a 2D circular annulus of inner radius a and outer radius b.

Eigenfunctions (Section III of thesis):

    ψ_{m,n}(r,φ) = C [Y_m(k_{m,n}a) J_m(k_{m,n}r) - J_m(k_{m,n}a) Y_m(k_{m,n}r)] e^{imφ}

Eigenvalues k_{m,n} are the nth roots of:

    J_m(ka) Y_m(kb) - J_m(kb) Y_m(ka) = 0

Thin-shell energy eigenvalues:

    E_{m,n} = (ℏ²/2m₀) [(nπ/aδ)² + (m/a)² - (1/2a)²] + O(δ)

Note: The third term -(1/2a)² is the pseudo-potential — a purely quantum
mechanical effect arising from the curvature of the annulus.

Usage
-----
    python circular_annulus.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, yv       # Bessel functions J_m, Y_m
from scipy.optimize import brentq

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "font.size": 12,
})

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────
a = 0.6   # inner radius
b = 1.0   # outer radius

Nr   = 300  # radial grid resolution
Nphi = 300  # azimuthal grid resolution

# ─────────────────────────────────────────────
# Find eigenvalues k_{m,n}
# ─────────────────────────────────────────────
def secular(k, m, a, b):
    """Secular equation whose roots give the eigenvalues k_{m,n}."""
    return jv(m, k*a) * yv(m, k*b) - jv(m, k*b) * yv(m, k*a)

def find_eigenvalues(m, a, b, n_roots=3, k_min=0.1, k_max=60.0, n_scan=5000):
    """Find the first n_roots eigenvalues for azimuthal order m."""
    k_vals  = np.linspace(k_min, k_max, n_scan)
    f_vals  = secular(k_vals, m, a, b)
    roots   = []
    for i in range(len(k_vals) - 1):
        if f_vals[i] * f_vals[i+1] < 0:
            try:
                root = brentq(secular, k_vals[i], k_vals[i+1], args=(m, a, b))
                roots.append(root)
            except ValueError:
                pass
        if len(roots) == n_roots:
            break
    return np.array(roots)

# ─────────────────────────────────────────────
# Radial eigenfunction R_{m,n}(r)
# ─────────────────────────────────────────────
def R_mn(r, k, m, a):
    """Unnormalised radial eigenfunction."""
    return yv(m, k*a) * jv(m, k*r) - jv(m, k*a) * yv(m, k*r)

def normalise(r, R):
    """Numerical normalisation over the annular domain."""
    norm = np.trapz(R**2 * r, r)
    return R / np.sqrt(norm) if norm > 0 else R

# ─────────────────────────────────────────────
# Build 2D polar grid
# ─────────────────────────────────────────────
r   = np.linspace(a, b, Nr)
phi = np.linspace(0, 2*np.pi, Nphi)
R_grid, Phi_grid = np.meshgrid(r, phi)
X_grid = R_grid * np.cos(Phi_grid)
Y_grid = R_grid * np.sin(Phi_grid)

# ─────────────────────────────────────────────
# Cases to plot: (m, n_index) pairs
# ─────────────────────────────────────────────
# n=1 and n=2 for any m (|ψ|² is independent of m for circular annulus)
cases = [(0, 0), (0, 1)]   # (m, n-1 index)
labels = [r'$(m,1)$', r'$(m,2)$']

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

for idx, ((m, ni), label) in enumerate(zip(cases, labels)):
    eigenvalues = find_eigenvalues(m, a, b, n_roots=ni+2)
    if len(eigenvalues) <= ni:
        print(f"Could not find eigenvalue for m={m}, n={ni+1}")
        continue
    k_mn = eigenvalues[ni]

    # Radial wavefunction
    R_r = R_mn(r, k_mn, m, a)
    R_r = normalise(r, R_r)

    # |ψ|² is independent of φ for the circular annulus
    R_2D = np.interp(R_grid.ravel(), r, R_r).reshape(R_grid.shape)
    prob  = R_2D**2   # azimuthal part integrates out

    ax = axes[idx]
    cf = ax.contourf(X_grid, Y_grid, prob, levels=60, cmap='jet')
    plt.colorbar(cf, ax=ax, label=r'$|\psi_{m,n}|^2$')
    ax.set_xlabel(r'$x$', fontsize=14)
    ax.set_ylabel(r'$y$', fontsize=14)
    ax.set_title(label + rf', $k_{{m,n}}={k_mn:.4f}$', fontsize=13)
    ax.set_aspect('equal')

fig.suptitle(rf'Probability Densities $|\psi_{{m,n}}|^2$ for Circular Annulus'
             rf' ($a={a}$, $b={b}$)', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/circular_annulus_probability_density.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# Thin-shell energy eigenvalues including pseudo-potential
# ─────────────────────────────────────────────
delta = (b - a) / a
print(f"\nThin-shell energy eigenvalues (ℏ²/2m₀=1, a={a}, b={b}, δ={delta:.4f}):")
print(f"{'(m,n)':<12} {'E_{m,n}':>14} {'Pseudo-pot term':>18}")
print("-" * 46)
for m in range(0, 3):
    for n in range(1, 4):
        confinement  = (n * np.pi / (a * delta))**2
        azimuthal    = (m / a)**2
        pseudo       = -(1 / (2*a))**2
        E            = confinement + azimuthal + pseudo
        print(f"({m},{n}){'':8} {E:>14.4f} {pseudo:>18.6f}")
