#!/usr/bin/env python
# coding: utf-8
"""
helical_annulus.py
==================
Computes and plots probability density functions for a quantum particle
confined in a 3D (elliptic) helical annulus, as described in Section VI
of the thesis.

Coordinate System (Section V-A, with h ≠ 0):
    x = d*cosh(ξ)*cos(η) + ρ*sinh(ξ)*cos(η) / sqrt(sinh²(ξ)+sin²(η))
    y = d*sinh(ξ)*sin(η) + ρ*cosh(ξ)*sin(η) / sqrt(sinh²(ξ)+sin²(η))
    z = h * η

Boundary Conditions (s spiral turns):
    ψ(ρ, η) = ψ(ρ, η + 2πs)

Modified Variational Basis:
    φ_{t,n}(ρ,η) = {cos(tη/s) or sin(tη/s)} × sin(nπ(ρ+ρ₀)/(2ρ₀))

Two experiments are produced:
  1. Probability density |ψ|² for representative eigenstates (3D surface plot
     and 2D projection).
  2. Variation of energy with pitch h, reproducing Figure 11 of the thesis.

Usage
-----
    python helical_annulus.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from scipy.linalg import eigh

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "font.size": 12,
})

# ─────────────────────────────────────────────
# Coordinate system
# ─────────────────────────────────────────────
def helix_coords(rho, eta, d, xi, h):
    """Map (ρ, η) → (x, y, z) for elliptic helical annulus."""
    sinh_xi = np.sinh(xi)
    cosh_xi = np.cosh(xi)
    denom   = np.sqrt(sinh_xi**2 + np.sin(eta)**2)
    x = d * cosh_xi * np.cos(eta) + rho * sinh_xi * np.cos(eta) / denom
    y = d * sinh_xi * np.sin(eta) + rho * cosh_xi * np.sin(eta) / denom
    z = h * eta
    return x, y, z

def metric_helix(rho, eta, d, xi, h, eps=1e-5):
    """Metric components g_ρρ and g_ηη for the helical annulus."""
    x0, y0, z0 = helix_coords(rho, eta, d, xi, h)
    xr, yr, zr = helix_coords(rho + eps, eta, d, xi, h)
    xe, ye, ze = helix_coords(rho, eta + eps, d, xi, h)
    g_rr = ((xr-x0)**2 + (yr-y0)**2 + (zr-z0)**2) / eps**2
    g_ee = ((xe-x0)**2 + (ye-y0)**2 + (ze-z0)**2) / eps**2
    return np.maximum(g_rr, 1e-12), np.maximum(g_ee, 1e-12)

# ─────────────────────────────────────────────
# Variational basis (modified for s turns)
# ─────────────────────────────────────────────
def basis_helix(rho, eta, t, n, rho0, s=1, symmetry='cos'):
    """Basis function φ_{t,n} for helical annulus with s spiral turns."""
    if symmetry == 'cos':
        ang = np.cos(t * eta / s) if t > 0 else np.ones_like(eta)
    else:
        ang = np.sin(t * eta / s) if t > 0 else np.zeros_like(eta)
    rad = np.sin(n * np.pi * (rho + rho0) / (2 * rho0))
    return ang * rad

def build_helix_matrices(d, xi, rho0, h, s=1, t_max=3, n_max=2,
                          N_rho=30, N_eta=60):
    """Build H and S matrices for the helical annulus variational problem."""
    eta_max  = 2 * np.pi * s
    rho_vals = np.linspace(-rho0, rho0, N_rho)
    eta_vals = np.linspace(0, eta_max, N_eta, endpoint=False)
    drho     = rho_vals[1] - rho_vals[0]
    deta     = eta_vals[1] - eta_vals[0]
    RHO, ETA = np.meshgrid(rho_vals, eta_vals, indexing='ij')

    GRR, GEE = metric_helix(RHO, ETA, d, xi, h)
    sqrtG    = np.sqrt(GRR * GEE)

    # Build basis
    basis = []
    for t in range(0, t_max + 1):
        for n in range(1, n_max + 1):
            basis.append((t, n, 'cos'))
            if t > 0:
                basis.append((t, n, 'sin'))
    N = len(basis)

    PHI = np.zeros((N, N_rho, N_eta))
    for i, (t, n, sym) in enumerate(basis):
        PHI[i] = basis_helix(RHO, ETA, t, n, rho0, s=s, symmetry=sym)

    def laplacian(psi):
        coeff_r  = sqrtG / GRR
        d_psi_r  = np.zeros_like(psi)
        d_psi_r[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2 * drho)
        flux_r   = coeff_r * d_psi_r
        div_r    = np.zeros_like(psi)
        div_r[1:-1, :] = (flux_r[2:, :] - flux_r[:-2, :]) / (2 * drho)
        coeff_e  = sqrtG / GEE
        d_psi_e  = np.gradient(psi, deta, axis=1)
        flux_e   = coeff_e * d_psi_e
        div_e    = np.gradient(flux_e, deta, axis=1)
        return (div_r + div_e) / sqrtG

    H = np.zeros((N, N))
    S = np.zeros((N, N))
    for i in range(N):
        lap_i = laplacian(PHI[i])
        for j in range(N):
            H[i, j] = np.sum(PHI[j] * (-0.5 * lap_i) * sqrtG) * drho * deta
            S[i, j] = np.sum(PHI[j] * PHI[i] * sqrtG) * drho * deta

    H = 0.5 * (H + H.T)
    S = 0.5 * (S + S.T)
    return H, S, basis, rho_vals, eta_vals, PHI

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────
d    = 1.59
eps  = 0.75
xi   = np.arccosh(1.0 / eps)
rho0 = 0.30
h    = 0.1     # pitch
s    = 2       # number of spiral turns

print(f"Helical Annulus Parameters:")
print(f"  d={d}, ε={eps}, ξ={xi:.4f}, ρ₀={rho0}, h={h}, s={s} turns")
print(f"\nBuilding variational matrices...")

H, S, basis, rho_vals, eta_vals, PHI = build_helix_matrices(
    d, xi, rho0, h, s=s, t_max=3, n_max=2, N_rho=30, N_eta=60
)
eigenvalues, eigenvectors = eigh(H, S)
print(f"  Lowest 4 eigenvalues: {eigenvalues[:4]}")

# ─────────────────────────────────────────────
# Reconstruct wavefunctions and map to 3D
# ─────────────────────────────────────────────
RHO_g, ETA_g = np.meshgrid(rho_vals, eta_vals, indexing='ij')
X_g, Y_g, Z_g = helix_coords(RHO_g, ETA_g, d, xi, h)

def reconstruct(coeffs, PHI):
    psi = np.zeros_like(PHI[0])
    for c, phi in zip(coeffs, PHI):
        psi += c * phi
    return psi

# ─────────────────────────────────────────────
# Plot |ψ|² for first 4 eigenstates (3D surface)
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), dpi=150)
cmap = plt.cm.jet

for idx in range(min(4, len(eigenvalues))):
    coeffs = eigenvectors[:, idx]
    psi    = reconstruct(coeffs, PHI)
    prob   = psi**2
    # Normalise for colour mapping
    prob_norm = (prob - prob.min()) / (prob.max() - prob.min() + 1e-12)
    colors     = cmap(prob_norm)

    ax = fig.add_subplot(2, 2, idx+1, projection='3d')
    ax.plot_surface(X_g, Y_g, Z_g, facecolors=colors,
                    rstride=1, cstride=1, alpha=0.9, linewidth=0)
    ax.set_xlabel(r'$x$', fontsize=11)
    ax.set_ylabel(r'$y$', fontsize=11)
    ax.set_zlabel(r'$z$', fontsize=11)
    ax.set_title(rf'State {idx+1}, $E={eigenvalues[idx]:.4f}$', fontsize=11)

fig.suptitle(rf'$|\psi|^2$ for Helical Annulus ($d={d}$, $\varepsilon={eps}$,'
             rf' $h={h}$, $s={s}$)', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/helical_annulus_probability_density.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# Energy vs pitch h (reproduces Figure 11)
# ─────────────────────────────────────────────
print("\nComputing energy vs pitch (reproducing Figure 11)...")
h_vals    = np.linspace(0.05, 6.0, 18)
eps_fig11 = 0.89
xi_fig11  = np.arccosh(1.0 / eps_fig11)
n_states_track = min(6, len(basis))
E_vs_h    = np.zeros((len(h_vals), n_states_track))

for hi, hv in enumerate(h_vals):
    print(f"  h = {hv:.2f}...")
    H_, S_, basis_, _, _, PHI_ = build_helix_matrices(
        d, xi_fig11, rho0, hv, s=2, t_max=2, n_max=1, N_rho=25, N_eta=50
    )
    evals, _ = eigh(H_, S_)
    E_vs_h[hi, :] = evals[:n_states_track]

fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
state_labels_fig11 = [rf'State {i+1}' for i in range(n_states_track)]
for i in range(n_states_track):
    ax.plot(h_vals, E_vs_h[:, i], label=state_labels_fig11[i], linewidth=1.5)

ax.set_xlabel(r'$h$ (pitch)', fontsize=16)
ax.set_ylabel(r'Energy', fontsize=16)
ax.legend(fontsize=11)
ax.grid(True)
fig.suptitle(rf'Energy vs Pitch for Elliptic Helical Annulus'
             rf' ($d={d}$, $\varepsilon={eps_fig11}$)', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/helical_annulus_energy_vs_pitch.pdf', format='pdf', bbox_inches='tight')
plt.show()
