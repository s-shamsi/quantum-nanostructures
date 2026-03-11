#!/usr/bin/env python
# coding: utf-8
"""
elliptical_annulus.py
=====================
Computes and plots probability density functions for a quantum particle
confined in a 2D elliptical annulus using the variational method described
in Section V of the thesis.

Coordinate System (Section V-A):
    x = d*cosh(ξ)*cos(η) + ρ*sinh(ξ)*cos(η) / sqrt(sinh²(ξ)+sin²(η))
    y = d*sinh(ξ)*sin(η) + ρ*cosh(ξ)*sin(η) / sqrt(sinh²(ξ)+sin²(η))

Parameters:
    d  — focal distance
    ξ  — eccentricity parameter (ε = 1/cosh(ξ))
    η  — angular coordinate ∈ [0, 2π]
    ρ  — radial displacement ∈ [-ρ₀, ρ₀]

Variational Basis (Section V-B):
    φ_{t,n}(ρ,η) = {cos(tη) or sin(tη)} × sin(nπ(ρ+ρ₀)/(2ρ₀))

Eigenvalue Problem:
    H a = E S a
    where H_{mn} = ∫ φ*_m Ĥ φ_n d²r
          S_{mn} = ∫ φ*_m φ_n d²r

The pseudo-potential U(η) = (1/4)[K(η)/2]² arises from the position-dependent
curvature of the ellipse and causes localisation of |ψ|² near the bends.

Usage
-----
    python elliptical_annulus.py
"""

import numpy as np
import os
os.makedirs("outputs", exist_ok=True)
import matplotlib.pyplot as plt
from scipy.linalg import eigh

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "font.size": 12,
})

# ─────────────────────────────────────────────
# Coordinate system
# ─────────────────────────────────────────────
def elliptic_coords(rho, eta, d, xi):
    """
    Map (ρ, η) → (x, y) using the custom elliptic coordinate system.
    ξ is fixed (determines eccentricity ε = 1/cosh(ξ)).
    """
    sinh_xi = np.sinh(xi)
    cosh_xi = np.cosh(xi)
    denom   = np.sqrt(sinh_xi**2 + np.sin(eta)**2)
    x = d * cosh_xi * np.cos(eta) + rho * sinh_xi * np.cos(eta) / denom
    y = d * sinh_xi * np.sin(eta) + rho * cosh_xi * np.sin(eta) / denom
    return x, y

def metric_components(rho, eta, d, xi, eps=1e-6):
    """
    Compute metric tensor components g_ρρ and g_ηη via finite differences.
    The coordinates are orthogonal by construction so g_ρη = 0.
    """
    x0, y0 = elliptic_coords(rho, eta, d, xi)

    # Partial derivatives w.r.t. ρ
    xr, yr = elliptic_coords(rho + eps, eta, d, xi)
    dxdr   = (xr - x0) / eps
    dydr   = (yr - y0) / eps

    # Partial derivatives w.r.t. η
    xe, ye = elliptic_coords(rho, eta + eps, d, xi)
    dxde   = (xe - x0) / eps
    dyde   = (ye - y0) / eps

    g_rr = dxdr**2 + dydr**2   # = 1 by construction
    g_ee = dxde**2 + dyde**2
    return g_rr, g_ee

# ─────────────────────────────────────────────
# Pseudo-potential from curvature (Section V-C)
# ─────────────────────────────────────────────
def local_curvature(eta, d, xi):
    """
    Local curvature K(η) of the ellipse at angular position η.
    K = (ab) / (a²sin²η + b²cos²η)^(3/2) for standard ellipse axes
    derived from the parameterisation.
    """
    sinh_xi = np.sinh(xi)
    cosh_xi = np.cosh(xi)
    # Semi-axes from the focal distance and xi
    A = d * cosh_xi   # semi-major axis
    B = d * sinh_xi   # semi-minor axis
    num   = A * B
    denom = (A**2 * np.sin(eta)**2 + B**2 * np.cos(eta)**2)**1.5
    return num / denom

def pseudo_potential(eta, d, xi):
    """
    U(η) = (K(η)/2)²/4  — confining pseudo-potential from curvature.
    For circular annulus K = 1/a = constant → U = 1/(4a²).
    """
    K = local_curvature(eta, d, xi)
    return (K / 2)**2 / 4.0

# ─────────────────────────────────────────────
# Variational method
# ─────────────────────────────────────────────
def basis_func(rho, eta, t, n, rho0, symmetry='cos'):
    """
    Variational basis function φ_{t,n}(ρ,η).
    symmetry = 'cos' or 'sin' (cos/sin tη factor).
    """
    if symmetry == 'cos':
        ang = np.cos(t * eta) if t > 0 else np.ones_like(eta)
    else:
        ang = np.sin(t * eta) if t > 0 else np.zeros_like(eta)
    rad = np.sin(n * np.pi * (rho + rho0) / (2 * rho0))
    return ang * rad

def build_matrices(d, xi, rho0, t_max=4, n_max=3, N_rho=40, N_eta=80):
    """
    Build Hamiltonian H and overlap S matrices via numerical integration.
    Uses finite-difference Laplacian in (ρ, η) coordinates.
    Returns eigenvalues and eigenvectors.
    """
    rho_vals = np.linspace(-rho0, rho0, N_rho)
    eta_vals = np.linspace(0, 2*np.pi, N_eta, endpoint=False)
    drho     = rho_vals[1] - rho_vals[0]
    deta     = eta_vals[1] - eta_vals[0]
    RHO, ETA = np.meshgrid(rho_vals, eta_vals, indexing='ij')

    # Metric on the full grid
    GRR, GEE = metric_components(RHO, ETA, d, xi)
    sqrtG    = np.sqrt(GRR * GEE)     # sqrt(det g) = sqrt(g_ρρ * g_ηη)

    # Build basis index list: (t, n, symmetry)
    basis = []
    for t in range(0, t_max + 1):
        for n in range(1, n_max + 1):
            basis.append((t, n, 'cos'))
            if t > 0:
                basis.append((t, n, 'sin'))
    N = len(basis)

    # Evaluate basis functions on grid
    PHI = np.zeros((N, N_rho, N_eta))
    for i, (t, n, sym) in enumerate(basis):
        PHI[i] = basis_func(RHO, ETA, t, n, rho0, sym)

    # Finite-difference Laplacian in curvilinear coords:
    # ∇²ψ ≈ (1/√g)[∂_ρ(√g/g_ρρ ∂_ρ ψ) + ∂_η(√g/g_ηη ∂_η ψ)]
    def laplacian(psi):
        lap = np.zeros_like(psi)
        # ρ contribution (interior only)
        coeff_rho = sqrtG / GRR
        d_psi_rho = np.zeros_like(psi)
        d_psi_rho[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * drho)
        flux_rho             = coeff_rho * d_psi_rho
        div_rho              = np.zeros_like(psi)
        div_rho[1:-1, :] = (flux_rho[2:, :] - flux_rho[:-2, :]) / (2 * drho)
        # η contribution (periodic)
        coeff_eta = sqrtG / GEE
        d_psi_eta = np.gradient(psi, deta, axis=1)
        flux_eta  = coeff_eta * d_psi_eta
        div_eta   = np.gradient(flux_eta, deta, axis=1)
        lap = (div_rho + div_eta) / sqrtG
        return lap

    H = np.zeros((N, N))
    S = np.zeros((N, N))

    for i in range(N):
        lap_i = laplacian(PHI[i])
        for j in range(N):
            integrand_H = PHI[j] * (-0.5 * lap_i) * sqrtG
            integrand_S = PHI[j] * PHI[i] * sqrtG
            H[i, j] = np.sum(integrand_H) * drho * deta
            S[i, j] = np.sum(integrand_S) * drho * deta

    # Symmetrise
    H = 0.5 * (H + H.T)
    S = 0.5 * (S + S.T)
    return H, S, basis, rho_vals, eta_vals, PHI

def reconstruct_wavefunction(coeffs, PHI):
    """Reconstruct ψ from variational coefficients."""
    psi = np.zeros_like(PHI[0])
    for c, phi in zip(coeffs, PHI):
        psi += c * phi
    return psi

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────
d    = 1.59   # focal distance
eps  = 0.89   # eccentricity
xi   = np.arccosh(1.0 / eps)  # ξ from eccentricity
rho0 = 0.30   # half-thickness of annulus (δ = 2ρ₀)
delta = 2 * rho0

print(f"Elliptical Annulus Parameters:")
print(f"  d = {d}, eccentricity ε = {eps:.2f}, ξ = {xi:.4f}")
print(f"  Thickness δ = {delta:.2f}, ρ₀ = {rho0:.2f}")
print(f"\nBuilding variational matrices (this may take a moment)...")

H, S, basis, rho_vals, eta_vals, PHI = build_matrices(
    d, xi, rho0, t_max=3, n_max=2, N_rho=35, N_eta=70
)

print(f"  Basis size: {len(basis)} functions")
print(f"  Solving generalised eigenvalue problem...")

# Solve generalised eigenvalue problem Ha = ESa
eigenvalues, eigenvectors = eigh(H, S)

print(f"  Lowest 6 eigenvalues: {eigenvalues[:6]}")

# ─────────────────────────────────────────────
# Map back to Cartesian for plotting
# ─────────────────────────────────────────────
RHO_g, ETA_g = np.meshgrid(rho_vals, eta_vals, indexing='ij')
X_g, Y_g = elliptic_coords(RHO_g, ETA_g, d, xi)

# ─────────────────────────────────────────────
# Plot first 6 eigenstates
# ─────────────────────────────────────────────
n_states = min(6, len(eigenvalues))
fig, axes = plt.subplots(3, 2, figsize=(13, 14), dpi=150)
axes = axes.flatten()

for idx in range(n_states):
    coeffs = eigenvectors[:, idx]
    psi    = reconstruct_wavefunction(coeffs, PHI)
    prob   = psi**2
    t_b, n_b, sym_b = basis[np.argmax(np.abs(coeffs))]

    ax = axes[idx]
    cf = ax.contourf(X_g, Y_g, prob, levels=50, cmap='jet')
    plt.colorbar(cf, ax=ax, label=r'$|\psi|^2$')
    ax.set_xlabel(r'$x$', fontsize=13)
    ax.set_ylabel(r'$y$', fontsize=13)
    ax.set_title(rf'State {idx+1}, $E={eigenvalues[idx]:.4f}$ (dominant: $t={t_b}$, $n={n_b}$, {sym_b})',
                 fontsize=10)
    ax.set_aspect('equal')

fig.suptitle(rf'Probability Densities $|\psi|^2$ for Elliptical Annulus'
             rf' ($d={d}$, $\varepsilon={eps}$, $\delta={delta:.2f}$)', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/elliptical_annulus_probability_density.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# Plot pseudo-potential U(η) for varying eccentricity
# ─────────────────────────────────────────────
eta_fine = np.linspace(0, 2*np.pi, 500)
eccentricities = [0.00, 0.50, 0.75, 0.89]
labels_eps = [rf'$\varepsilon = {e:.2f}$' for e in eccentricities]

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
for eps_val, lbl in zip(eccentricities, labels_eps):
    if eps_val == 0.0:
        # Circle: U = const = 1/(4a²)
        a_circ = d   # approximate radius
        U = np.full_like(eta_fine, -1.0 / (4 * a_circ**2))
    else:
        xi_val = np.arccosh(1.0 / eps_val)
        U = -pseudo_potential(eta_fine, d, xi_val)
    ax.plot(eta_fine, U, label=lbl, linewidth=1.5)

ax.set_xlabel(r'$\eta$', fontsize=16)
ax.set_ylabel(r'$U(\eta)$', fontsize=16)
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
ax.legend(fontsize=12)
ax.grid(True)
fig.suptitle(r'Pseudo-potential $U(\eta)$ vs Angular Position for Varying Eccentricity',
             fontsize=13)
plt.tight_layout()
plt.savefig('outputs/elliptical_annulus_pseudopotential.pdf', format='pdf', bbox_inches='tight')
plt.show()
