#!/usr/bin/env python
# coding: utf-8
"""
spherical_shell.py
==================
Computes and plots probability density functions for a quantum particle
confined in a 3D spherical shell of inner radius a and outer radius b.

Eigenfunctions (Section IV of thesis):

    ψ_{n,l,m}(r,θ,φ) = C [y_l(k a) j_l(k r) - j_l(k a) y_l(k r)] P_l^m(cosθ) e^{imφ}

where j_l and y_l are spherical Bessel functions of order l.

Eigenvalues k_{n,l} satisfy:

    j_l(ka) y_l(kb) - j_l(kb) y_l(ka) = 0

Key result: The pseudo-potential U = 0 for the spherical shell because R₁ = R₂ = a.

Thin-shell energy eigenvalues:

    E_{n,l} = (ℏ²/2m₀) [(nπ/δa)² + l(l+1)/a²]

Plots are 2D projections of |ψ|² showing r and θ dependence (φ integrated out),
reproducing Figure 4 of the thesis.

Usage
-----
    python spherical_shell.py
"""

import numpy as np
import os
os.makedirs("outputs", exist_ok=True)
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn, lpmv
from scipy.optimize import brentq

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "font.size": 12,
})

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────
a = 1.0   # inner radius
b = 1.5   # outer radius

Nr     = 200   # radial resolution
Ntheta = 200   # polar resolution

# ─────────────────────────────────────────────
# Find eigenvalues k_{n,l}
# ─────────────────────────────────────────────
def secular(k, l, a, b):
    """Secular equation for spherical shell eigenvalues."""
    return (spherical_jn(l, k*a) * spherical_yn(l, k*b)
          - spherical_jn(l, k*b) * spherical_yn(l, k*a))

def find_eigenvalues(l, a, b, n_roots=3, k_min=0.5, k_max=80.0, n_scan=8000):
    """Find the first n_roots radial eigenvalues for angular momentum l."""
    k_vals = np.linspace(k_min, k_max, n_scan)
    f_vals = secular(k_vals, l, a, b)
    roots  = []
    for i in range(len(k_vals) - 1):
        if np.isfinite(f_vals[i]) and np.isfinite(f_vals[i+1]):
            if f_vals[i] * f_vals[i+1] < 0:
                try:
                    root = brentq(secular, k_vals[i], k_vals[i+1], args=(l, a, b))
                    roots.append(root)
                except ValueError:
                    pass
        if len(roots) == n_roots:
            break
    return np.array(roots)

# ─────────────────────────────────────────────
# Radial and angular parts
# ─────────────────────────────────────────────
def R_nl(r, k, l, a):
    """Unnormalised radial eigenfunction."""
    return (spherical_yn(l, k*a) * spherical_jn(l, k*r)
          - spherical_jn(l, k*a) * spherical_yn(l, k*r))

def normalise_radial(r, R):
    norm = np.trapz(R**2 * r**2, r)
    return R / np.sqrt(norm) if norm > 0 else R

def P_lm(l, m, theta):
    """Associated Legendre polynomial P_l^m(cos θ)."""
    x = np.cos(theta)
    return lpmv(m, l, x)

# ─────────────────────────────────────────────
# Build (r, θ) grid — 2D cross section at fixed φ
# ─────────────────────────────────────────────
r     = np.linspace(a, b, Nr)
theta = np.linspace(0, np.pi, Ntheta)
R_g, Th_g = np.meshgrid(r, theta)

# Convert to Cartesian for plotting
X_plot = R_g * np.sin(Th_g)   # x = r sinθ
Z_plot = R_g * np.cos(Th_g)   # z = r cosθ

# ─────────────────────────────────────────────
# Cases: (n_index, l, m) — matching Figure 4 of thesis
# ─────────────────────────────────────────────
cases = [
    (0, 0, 0),   # (1,0,0)
    (0, 1, 0),   # (1,1,0)
    (0, 1, 1),   # (1,1,1)
    (1, 2, 0),   # (2,2,0)
    (1, 2, 1),   # (2,2,1)
    (1, 2, 2),   # (2,2,2)
]
labels = [
    r'$(n,\ell,m)=(1,0,0)$', r'$(n,\ell,m)=(1,1,0)$',
    r'$(n,\ell,m)=(1,1,1)$', r'$(n,\ell,m)=(2,2,0)$',
    r'$(n,\ell,m)=(2,2,1)$', r'$(n,\ell,m)=(2,2,2)$',
]

fig, axes = plt.subplots(3, 2, figsize=(11, 14), dpi=150)
axes = axes.flatten()

for idx, ((ni, l, m), label) in enumerate(zip(cases, labels)):
    eigenvalues = find_eigenvalues(l, a, b, n_roots=ni+2)
    if len(eigenvalues) <= ni:
        print(f"Could not find eigenvalue for l={l}, n_index={ni}")
        continue
    k = eigenvalues[ni]

    # Radial part
    R_r  = R_nl(r, k, l, a)
    R_r  = normalise_radial(r, R_r)
    R_2D = np.interp(R_g.ravel(), r, R_r).reshape(R_g.shape)

    # Angular part — P_l^m(cosθ)
    P_th = P_lm(l, m, theta)
    # Normalise angular part
    norm_ang = np.trapz(P_th**2 * np.sin(theta), theta)
    if norm_ang > 0:
        P_th /= np.sqrt(norm_ang)
    P_2D = P_th[np.newaxis, :].T  # broadcast over r

    # |ψ|² (φ-integrated, so the e^{imφ} factor gives a constant)
    prob = (R_2D * P_2D)**2

    ax = axes[idx]
    # Reflect to full circle (θ: 0→π, so plot both hemispheres)
    X_full = np.concatenate([-X_plot[::-1, :], X_plot], axis=0)
    Z_full = np.concatenate([ Z_plot[::-1, :], Z_plot], axis=0)
    prob_full = np.concatenate([prob[::-1, :], prob], axis=0)

    cf = ax.contourf(X_full, Z_full, prob_full, levels=60, cmap='jet')
    plt.colorbar(cf, ax=ax, label=r'$|\psi|^2$')
    ax.set_xlabel(r'$x$', fontsize=13)
    ax.set_ylabel(r'$z$', fontsize=13)
    ax.set_title(label + rf', $k={k:.4f}$', fontsize=11)
    ax.set_aspect('equal')

fig.suptitle(rf'Probability Densities $|\psi_{{n,\ell,m}}|^2$ for Spherical Shell'
             rf' ($a={a}$, $b={b}$)', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/spherical_shell_probability_density.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# Thin-shell energy eigenvalues (no pseudo-potential: U=0)
# ─────────────────────────────────────────────
delta = (b - a) / a
print(f"\nThin-shell energy eigenvalues for Spherical Shell (ℏ²/2m₀=1):")
print(f"Note: Pseudo-potential U = (1/R1 - 1/R2)² / 4 = 0  (R1 = R2 = a)")
print(f"{'(n,l)':<12} {'E_{n,l}':>12}")
print("-" * 26)
for n in range(1, 3):
    for l in range(0, 4):
        E = (n * np.pi / (delta * a))**2 + l*(l+1) / a**2
        print(f"({n},{l}){'':8} {E:>12.4f}")
