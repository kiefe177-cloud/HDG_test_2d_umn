"""
Plot the baseflow density and velocity quiver from bU.csv (interactive plotly).

Inputs (all in the same directory):
    bU.csv      : 5*(p+1)^2 rows by Ne cols. Stacked as [rho; rho*u; rho*v; rho*w; E].
    elem_x.csv  : (p+1)^2 rows by Ne cols. Per-element node x-coordinates.
    elem_y.csv  : (p+1)^2 rows by Ne cols. Per-element node y-coordinates.

Produces a single interactive figure with two side-by-side panels:
    Left  : density coloured by rho/rho_e
    Right : velocity quiver (every node, no background)

Pan / zoom / hover are enabled. Use the toolbar in the top-right of the
figure to switch zoom modes, save as PNG, etc.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# --- File paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / "output" / "cyl_driver"

bU_path = DATA_DIR / "bU.csv"
xN_path = DATA_DIR / "elem_x.csv"
yN_path = DATA_DIR / "elem_y.csv"

# --- Load data ---
bU = pd.read_csv(bU_path, header=None).values   # (5*npe, Ne)
xN = pd.read_csv(xN_path, header=None).values   # (npe, Ne)
yN = pd.read_csv(yN_path, header=None).values   # (npe, Ne)

Ne  = bU.shape[1]
npe = xN.shape[0]

assert bU.shape[0] == 5 * npe, (
    f"bU has {bU.shape[0]} rows, expected 5*npe = {5 * npe}."
)
assert xN.shape[1] == Ne and yN.shape[1] == Ne, (
    "Element counts mismatch between bU and coordinate files."
)

# --- Slice out conservative variables ---
rho  = bU[0 * npe : 1 * npe, :]
rhou = bU[1 * npe : 2 * npe, :]
rhov = bU[2 * npe : 3 * npe, :]

# Primitive velocities (defensive divide)
safe_rho = np.where(rho > 0, rho, 1.0)
u = rhou / safe_rho
v = rhov / safe_rho

# Flatten everything (column-major to match how the C++ flattened the field data)
x_flat   = xN.flatten(order="F")
y_flat   = yN.flatten(order="F")
rho_flat = rho.flatten(order="F")
u_flat   = u.flatten(order="F")
v_flat   = v.flatten(order="F")

print(f"Loaded {Ne} elements, {npe} nodes/element ({Ne * npe} total nodes)")
print(f"  rho range: [{rho_flat.min():.4e}, {rho_flat.max():.4e}]")
print(f"  |u| range: [{np.abs(u_flat).min():.4e}, {np.abs(u_flat).max():.4e}]")
print(f"  |v| range: [{np.abs(v_flat).min():.4e}, {np.abs(v_flat).max():.4e}]")

# --- Quiver settings ---
# Tweak these to taste:
#   STRIDE: draw every Nth node (1 = every arrow)
#   ARROW_SCALE: arrow length multiplier (bigger = longer arrows)
#   ARROWHEAD_SCALE: arrowhead size relative to shaft
STRIDE          = 10
ARROW_SCALE     = 0.40
ARROWHEAD_SCALE = 0.45

xs = x_flat[::STRIDE]
ys = y_flat[::STRIDE]
us = u_flat[::STRIDE]
vs = v_flat[::STRIDE]

# Build a quiver figure separately so we can pull out its trace and add it to a subplot.
quiv_fig = ff.create_quiver(
    xs, ys, us, vs,
    scale=ARROW_SCALE,
    arrow_scale=ARROWHEAD_SCALE,
    line=dict(width=1.2, color="black"),
    name="velocity",
)
quiver_trace = quiv_fig.data[0]

# --- Build the side-by-side figure ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Density (rho/rho_e)", f"Velocity quiver (every {STRIDE}th node)"),
    horizontal_spacing=0.12,
)

# Left panel: density scatter
fig.add_trace(
    go.Scattergl(
        x=x_flat, y=y_flat,
        mode="markers",
        marker=dict(
            size=4,
            color=rho_flat,
            colorscale="Viridis",
            colorbar=dict(title="rho/rho_e", x=0.46),
            showscale=True,
        ),
        hovertemplate=(
            "x = %{x:.4f}<br>"
            "y = %{y:.4f}<br>"
            "rho = %{marker.color:.4e}<extra></extra>"
        ),
        name="rho",
    ),
    row=1, col=1,
)

# Right panel: quiver only, every node
fig.add_trace(quiver_trace, row=1, col=2)

# Equal aspect ratio for both panels (so the cylinder doesn't squash)
fig.update_yaxes(scaleanchor="x",  scaleratio=1, row=1, col=1)
fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)

fig.update_xaxes(title_text="x / l_0", row=1, col=1)
fig.update_xaxes(title_text="x / l_0", row=1, col=2)
fig.update_yaxes(title_text="y / l_0", row=1, col=1)
fig.update_yaxes(title_text="y / l_0", row=1, col=2)

fig.update_layout(
    title=f"Baseflow ({Ne} elements, {npe} nodes/elem)",
    width=1500,
    height=750,
    showlegend=False,
)

fig.show()