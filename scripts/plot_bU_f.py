"""
Plot the face-trace baseflow density and velocity quiver from bU_f.csv (interactive plotly).

Inputs (all in the same directory):
    bU_f.csv     : 5*(p+1) rows by Nf cols. Stacked as [rho_f; rho*u_f; rho*v_f; rho*w_f; E_f].
    face_x.csv   : (p+1) rows by Nf cols. Per-face node x-coordinates.
    face_y.csv   : (p+1) rows by Nf cols. Per-face node y-coordinates.

(If your CSVs are still named elem_x_f.csv / elem_y_f.csv, change the paths below.)

Produces a single interactive figure with two side-by-side panels:
    Left  : density on faces, coloured by rho/rho_e
    Right : velocity quiver on faces (every Nth face node)

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

bU_path = DATA_DIR / "bU_f.csv"
xN_path = DATA_DIR / "face_x.csv"      # rename if your driver wrote elem_x_f.csv
yN_path = DATA_DIR / "face_y.csv"      # rename if your driver wrote elem_y_f.csv

# --- Load data ---
bU = pd.read_csv(bU_path, header=None).values   # (5*nf, Nf)
xN = pd.read_csv(xN_path, header=None).values   # (nf, Nf)
yN = pd.read_csv(yN_path, header=None).values   # (nf, Nf)

Nf = bU.shape[1]
nf = xN.shape[0]

assert bU.shape[0] == 5 * nf, (
    f"bU_f has {bU.shape[0]} rows, expected 5*nf = {5 * nf}. "
    f"Check that the coordinate files have the right number of nodes per face."
)
assert xN.shape[1] == Nf and yN.shape[1] == Nf, (
    "Face counts mismatch between bU_f and coordinate files."
)

# --- Slice out conservative variables ---
rho  = bU[0 * nf : 1 * nf, :]
rhou = bU[1 * nf : 2 * nf, :]
rhov = bU[2 * nf : 3 * nf, :]
# rhow = bU[3 * nf : 4 * nf, :]   # unused for 2D quiver
# E    = bU[4 * nf : 5 * nf, :]

# Primitive velocities (defensive divide)
safe_rho = np.where(rho > 0, rho, 1.0)
u = rhou / safe_rho
v = rhov / safe_rho

# Flatten (column-major to match how the C++ flattened the face data)
x_flat   = xN.flatten(order="F")
y_flat   = yN.flatten(order="F")
rho_flat = rho.flatten(order="F")
u_flat   = u.flatten(order="F")
v_flat   = v.flatten(order="F")

print(f"Loaded {Nf} faces, {nf} nodes/face ({Nf * nf} total face nodes)")
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
    subplot_titles=("Face density (rho/rho_e)", f"Face velocity quiver (every {STRIDE}th node)"),
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

# Right panel: quiver only
fig.add_trace(quiver_trace, row=1, col=2)

# Equal aspect ratio for both panels
fig.update_yaxes(scaleanchor="x",  scaleratio=1, row=1, col=1)
fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)

fig.update_xaxes(title_text="x / l_0", row=1, col=1)
fig.update_xaxes(title_text="x / l_0", row=1, col=2)
fig.update_yaxes(title_text="y / l_0", row=1, col=1)
fig.update_yaxes(title_text="y / l_0", row=1, col=2)

fig.update_layout(
    title=f"Face-trace baseflow ({Nf} faces, {nf} nodes/face)",
    width=1500,
    height=750,
    showlegend=False,
)

fig.show()