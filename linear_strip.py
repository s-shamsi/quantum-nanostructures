#!/usr/bin/env python
# coding: utf-8
"""
linear_strip.py
===============
Computes and plots the eigenfunctions and probability density functions
for a quantum particle confined in a 2D linear strip (rectangular infinite
square well) of length a and width b.

Analytical solution (Section II of thesis):

    ψ_{m,n}(x,y) = sqrt(4/ab) * sin(mπx/a) * sin(nπy/b)

Energy eigenvalues:

    E_{m,n} = (ℏ²/2m₀) * [(mπ/a)² + (nπ/b)²]

Quantum numbers m, n are positive integers.

Usage
-----
    python linear_strip.py

Note: Requires a LaTeX installation for plot rendering (text.usetex = True).
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "font.size": 12,
})

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────
a = 2.0   # length along x
b = 1.0   # width along y
Nx = 300  # grid resolution in x
Ny = 150  # grid resolution in y

x = np.linspace(0, a, Nx)
y = np.linspace(0, b, Ny)
X, Y = np.meshgrid(x, y)

# ─────────────────────────────────────────────
# Eigenfunction
# ─────────────────────────────────────────────
def psi(m, n, X, Y, a, b):
    """Normalised eigenfunction ψ_{m,n}(x,y)."""
    return np.sqrt(4 / (a * b)) * np.sin(m * np.pi * X / a) * np.sin(n * np.pi * Y / b)

def energy(m, n, a, b, hbar=1.0, m0=1.0):
    """Energy eigenvalue E_{m,n} in units where ℏ²/2m₀ = 1."""
    return (hbar**2 / (2 * m0)) * ((m * np.pi / a)**2 + (n * np.pi / b)**2)

# ─────────────────────────────────────────────
# Quantum number pairs to plot
# ─────────────────────────────────────────────
quantum_numbers = [(1, 1), (2, 1), (1, 2), (2, 2)]
titles = [r'$(m,n) = (1,1)$', r'$(m,n) = (2,1)$',
          r'$(m,n) = (1,2)$', r'$(m,n) = (2,2)$']

# ─────────────────────────────────────────────
# Plot Eigenfunctions ψ_{m,n}
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
axes = axes.flatten()

for idx, ((m, n), title) in enumerate(zip(quantum_numbers, titles)):
    psi_mn = psi(m, n, X, Y, a, b)
    E_mn   = energy(m, n, a, b)
    ax = axes[idx]
    cf = ax.contourf(X, Y, psi_mn, levels=40, cmap='RdBu_r')
    ax.contour(X, Y, psi_mn, levels=[0], colors='k', linewidths=0.8, linestyles='--')
    plt.colorbar(cf, ax=ax, label=r'$\psi_{m,n}$')
    ax.set_xlabel(r'$x$', fontsize=14)
    ax.set_ylabel(r'$y$', fontsize=14)
    ax.set_title(title + rf', $E = {E_mn:.3f}$', fontsize=13)
    ax.set_aspect('equal')

fig.suptitle(r'Eigenfunctions $\psi_{m,n}(x,y)$ of the Linear Strip'
             rf' ($a={a}$, $b={b}$)', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/linear_strip_eigenfunctions.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# Plot Probability Densities |ψ_{m,n}|²
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
axes = axes.flatten()

for idx, ((m, n), title) in enumerate(zip(quantum_numbers, titles)):
    prob = psi(m, n, X, Y, a, b)**2
    ax = axes[idx]
    cf = ax.contourf(X, Y, prob, levels=40, cmap='hot_r')
    plt.colorbar(cf, ax=ax, label=r'$|\psi_{m,n}|^2$')
    ax.set_xlabel(r'$x$', fontsize=14)
    ax.set_ylabel(r'$y$', fontsize=14)
    ax.set_title(title, fontsize=13)
    ax.set_aspect('equal')

fig.suptitle(r'Probability Densities $|\psi_{m,n}|^2$ of the Linear Strip'
             rf' ($a={a}$, $b={b}$)', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/linear_strip_probability_density.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# Print Energy Eigenvalues
# ─────────────────────────────────────────────
print("Energy Eigenvalues E_{m,n} (ℏ²/2m₀ = 1):")
print(f"{'(m,n)':<12} {'Energy':>10}")
print("-" * 24)
for m in range(1, 4):
    for n in range(1, 4):
        print(f"({m},{n}){'':8} {energy(m, n, a, b):>10.4f}")
