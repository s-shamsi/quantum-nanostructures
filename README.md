# Dynamics of Quantum Mechanical Particles in Exotic Nanostructures

Numerical implementation accompanying the BSc thesis:

> **Dynamics of Quantum Mechanical Particles in Exotic Nanostructures**
> Saad A. Shamsi — Blackett Laboratory, Imperial College London (2013)

---

## Overview

This project solves the Time-Independent Schrödinger Equation (TISE) for a
free quantum particle confined to a series of increasingly complex 2D and 3D
nanostructures. The central theme is the emergence of a **curvature-induced
pseudo-potential** — a purely quantum mechanical confining effect that arises
from the intrinsic geometry of curved nanostructures.

The key result is the general pseudo-potential formula:

$$U = \frac{1}{4}\left(\frac{1}{R_1} - \frac{1}{R_2}\right)^2$$

where $R_1$ and $R_2$ are the principal radii of curvature of the surface.

---

## Nanostructures & Scripts

| Script | Nanostructure | Method | Key Physics |
|---|---|---|---|
| `linear_strip.py` | Rectangular strip | Analytical | 2D infinite square well; no curvature |
| `circular_annulus.py` | Circular annulus | Bessel eigenvalue solver | Constant pseudo-potential $U = 1/(4a^2)$ |
| `spherical_shell.py` | Spherical shell | Spherical Bessel eigenvalue solver | $U = 0$ since $R_1 = R_2$ |
| `elliptical_annulus.py` | Elliptical annulus | Variational method | Position-dependent $U(\eta)$; curvature localisation |
| `helical_annulus.py` | Helical annulus | Variational method (3D) | All 2D cases recovered as degeneracies |

---

## Physics Summary

### Linear Strip
Analytical eigenfunctions:

$$\psi_{m,n}(x,y) = \sqrt{\frac{4}{ab}}\sin \left(\frac{m\pi x}{a}\right)\sin \left(\frac{n\pi y}{b}\right)$$

### Circular Annulus
Bessel function eigenstates with eigenvalues $k_{m,n}$ satisfying:

$$J_m(ka)Y_m(kb) - J_m(kb)Y_m(ka) = 0$$

Thin-shell energy (note the **pseudo-potential** third term):

$$E_{m,n} = \frac{\hbar^2}{2m_0}\left[\left(\frac{n\pi}{a\delta}\right)^2 + \left(\frac{m}{a}\right)^2 - \left(\frac{1}{2a}\right)^2\right]$$

### Spherical Shell
$U = 0$ because $R_1 = R_2 = a$ for the sphere. Energy:

$$E_{n,\ell} = \frac{\hbar^2}{2m_0}\left[\left(\frac{n\pi}{\delta a}\right)^2 + \frac{\ell(\ell+1)}{a^2}\right]$$

### Elliptical Annulus
Position-dependent curvature solved by the **variational method** with basis:

$$\phi_{t,n}(\rho,\eta) = \begin{cases}\cos(t\eta)\\\sin(t\eta)\end{cases}\sin \left(\frac{n\pi(\rho+\rho_0)}{2\rho_0}\right)$$

Probability density localises near bends (high curvature, deep pseudo-potential).

### Helical Annulus
Extends the elliptical annulus to 3D with pitch $h$. All 2D nanostructures
are recovered as degeneracies of this case. Modified basis uses $t\eta/s$
for $s$ spiral turns.

---

## Project Structure

```
quantum-nanostructures/
│
├── linear_strip.py           # Analytical: rectangular infinite square well
├── circular_annulus.py       # Bessel functions: circular annulus eigenvalues
├── spherical_shell.py        # Spherical Bessel + Legendre: spherical shell
├── elliptical_annulus.py     # Variational: elliptical annulus
├── helical_annulus.py        # Variational 3D: helical annulus + energy vs pitch
│
├── outputs/                  # Generated plots saved here
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Requirements

- Python 3.8+
- A working **LaTeX installation** for plot rendering (`text.usetex = True`)
  - Ubuntu/Debian: `sudo apt install texlive-full`
  - macOS: `brew install --cask mactex`
  - Windows: [MiKTeX](https://miktex.org/)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run each script from the project root:

```bash
python linear_strip.py
python circular_annulus.py
python spherical_shell.py
python elliptical_annulus.py    # ~1–2 min runtime (variational solver)
python helical_annulus.py       # ~5–10 min runtime (3D variational + pitch sweep)
```

All plots are saved as PDFs in the `outputs/` directory.

> **Note:** `elliptical_annulus.py` and `helical_annulus.py` involve
> numerical matrix assembly and may take a few minutes to run depending
> on your hardware. Grid resolution parameters (`N_rho`, `N_eta`) can
> be reduced in the scripts for faster (lower resolution) results.

---

## Outputs

| File | Description |
|---|---|
| `linear_strip_eigenfunctions.pdf` | Contour plots of $\psi_{m,n}$ |
| `linear_strip_probability_density.pdf` | Contour plots of $\|\psi_{m,n}\|^2$ |
| `circular_annulus_probability_density.pdf` | $\|\psi_{m,n}\|^2$ for $n=1,2$ |
| `spherical_shell_probability_density.pdf` | 6-panel $\|\psi_{n,\ell,m}\|^2$ cross-sections |
| `elliptical_annulus_probability_density.pdf` | Variational eigenstates |
| `elliptical_annulus_pseudopotential.pdf` | $U(\eta)$ vs $\eta$ for varying eccentricity |
| `helical_annulus_probability_density.pdf` | 3D surface plots of $\|\psi\|^2$ |
| `helical_annulus_energy_vs_pitch.pdf` | Energy vs pitch (Figure 11 of thesis) |

---

## Dependencies

- [`numpy`](https://numpy.org/) — array operations
- [`matplotlib`](https://matplotlib.org/) — plotting with LaTeX rendering
- [`scipy`](https://scipy.org/) — Bessel functions, Legendre polynomials, eigensolvers

---

## License

MIT License. See `LICENSE` for details.
