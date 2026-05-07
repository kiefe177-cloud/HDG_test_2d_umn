"""
Plot the diagnostics for a TFI/elliptic-smoothed cylinder mesh.

Reads cyl_grid.h5 produced by the C++ driver and produces four figures
matching MATLAB's plot_comp_tfi_grid:

  1. Computational domain (phi, psi) with the test point overlaid.
  2. Physical domain with grid lines and curved boundary walls.
  3. Skewness field (degrees from orthogonal) as a pcolormesh.
  4. Jacobian field as a pcolormesh.
"""

import h5py
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
H5_PATH    = SCRIPT_DIR.parent / 'output' / 'cyl_driver' /'cyl_grid.h5'

if not H5_PATH.exists():
    print(f"Error: {H5_PATH} not found. Run the cyl_driver first.")
    raise SystemExit(1)

with h5py.File(H5_PATH, 'r') as f:
    Phi   = f['computational/Phi'][:]
    Psi   = f['computational/Psi'][:]
    phi_p = f['computational/phi_p'][()]   # scalar
    psi_p = f['computational/psi_p'][()]   # scalar

    X   = f['physical/X'][:]
    Y   = f['physical/Y'][:]
    x_p = f['physical/x_p'][()]            # scalar
    y_p = f['physical/y_p'][()]            # scalar

    xl, yl = f['boundaries/left/x'][:],   f['boundaries/left/y'][:]
    xr, yr = f['boundaries/right/x'][:],  f['boundaries/right/y'][:]
    xb, yb = f['boundaries/bottom/x'][:], f['boundaries/bottom/y'][:]
    xt, yt = f['boundaries/top/x'][:],    f['boundaries/top/y'][:]

    skew = f['diagnostics/skewness'][:]
    J    = f['diagnostics/jacobian'][:]


# --- Figure 1: computational domain ---
fig1, ax1 = plt.subplots(figsize=(7, 7))
for i in range(Phi.shape[0]):
    ax1.plot(Phi[i, :], Psi[i, :], 'k-', linewidth=0.5)
for j in range(Phi.shape[1]):
    ax1.plot(Phi[:, j], Psi[:, j], 'k-', linewidth=0.5)
ax1.plot(phi_p, psi_p, 'ro', markersize=10, markeredgewidth=2,
         markerfacecolor='none')
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('Computational domain (phi, psi)')


# --- Figure 2: physical domain with curved walls ---
fig2, ax2 = plt.subplots(figsize=(8, 8))
for i in range(X.shape[0]):
    ax2.plot(X[i, :], Y[i, :], color=(0.5, 0.5, 0.5), linewidth=0.4)
for j in range(X.shape[1]):
    ax2.plot(X[:, j], Y[:, j], color=(0.5, 0.5, 0.5), linewidth=0.4)
ax2.plot(xl, yl, 'r-', linewidth=2, label='left/right walls')
ax2.plot(xr, yr, 'r-', linewidth=2)
ax2.plot(xb, yb, 'b-', linewidth=2, label='bottom/top walls')
ax2.plot(xt, yt, 'b-', linewidth=2)
ax2.plot(x_p, y_p, 'ro', markersize=8)
ax2.set_aspect('equal')
ax2.set_title('Physical domain')
ax2.legend(loc='best', fontsize=9)


# --- Figure 3: skewness field ---
fig3, ax3 = plt.subplots(figsize=(8, 7))
m3 = ax3.pcolormesh(X, Y, skew, shading='gouraud', cmap='viridis')
ax3.set_aspect('equal')
ax3.set_title('Grid skewness (degrees from orthogonal)')
fig3.colorbar(m3, ax=ax3)


# --- Figure 4: Jacobian field ---
fig4, ax4 = plt.subplots(figsize=(8, 7))
m4 = ax4.pcolormesh(X, Y, J, shading='gouraud', cmap='viridis')
ax4.set_aspect('equal')
ax4.set_title('Grid Jacobian')
fig4.colorbar(m4, ax=ax4)


plt.show()