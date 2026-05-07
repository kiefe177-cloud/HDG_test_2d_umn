"""
Plot the baseflow gradient fields bUx and bUy from CSV (interactive plotly).

Inputs (all in the same directory):
    bUx.csv     : Me rows by Ne cols, where Me = nvar*(p+1)^2.
                  Stacked variables: [d/dx rho; d/dx rhou; d/dx rhov; d/dx rhow; d/dx E].
    bUy.csv     : Me rows by Ne cols. Same stacking, y-derivatives.
    elem_x.csv  : (p+1)^2 rows by Ne cols. Per-element node x-coordinates.
    elem_y.csv  : (p+1)^2 rows by Ne cols. Per-element node y-coordinates.

Produces a side-by-side panel for one variable at a time. Pick which variable to
plot via the VAR_INDEX constant near the top:
    0 = rho
    1 = rho*u
    2 = rho*v
    3 = rho*w
    4 = E

Pan / zoom / hover are enabled. Use the toolbar in the top-right of the
figure to switch zoom modes, save as PNG, etc.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- File paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / "output" / "cyl_driver"

bUx_path = DATA_DIR / "bUx.csv"
bUy_path = DATA_DIR / "bUy.csv"
xN_path  = DATA_DIR / "elem_x.csv"
yN_path  = DATA_DIR / "elem_y.csv"

# Which conservative variable's gradient to display (0..nvar-1)
VAR_INDEX = 0   # 0 = rho

VAR_NAMES = ["rho", "rho*u", "rho*v", "rho*w", "E"]

# --- Load data ---
bUx = pd.read_csv(bUx_path, header=None).values   # (Me, Ne)
bUy = pd.read_csv(bUy_path, header=None).values   # (Me, Ne)
xN  = pd.read_csv(xN_path,  header=None).values   # (npe, Ne)
yN  = pd.read_csv(yN_path,  header=None).values   # (npe, Ne)

Ne  = bUx.shape[1]
Me  = bUx.shape[0]
npe = xN.shape[0]

# Recover nvar = Me / npe
if Me % npe != 0:
    raise ValueError(
        f"Me ({Me}) is not a multiple of npe ({npe}); "
        "check that bUx and elem_x come from the same run."
    )
nvar = Me // npe

assert bUy.shape == bUx.shape, "bUx and bUy shapes mismatch"
assert xN.shape[1] == Ne and yN.shape[1] == Ne, "Element counts mismatch"
assert 0 <= VAR_INDEX < nvar, f"VAR_INDEX {VAR_INDEX} outside [0, {nvar})"

# --- Slice out the requested variable's gradient ---
row_lo = VAR_INDEX * npe
row_hi = (VAR_INDEX + 1) * npe
bUx_var = bUx[row_lo:row_hi, :]   # (npe, Ne)
bUy_var = bUy[row_lo:row_hi, :]

# --- Flatten (column-major to match how the C++ flattened element data) ---
x_flat   = xN.flatten(order="F")
y_flat   = yN.flatten(order="F")
bUx_flat = bUx_var.flatten(order="F")
bUy_flat = bUy_var.flatten(order="F")

var_label = VAR_NAMES[VAR_INDEX] if VAR_INDEX < len(VAR_NAMES) else f"var{VAR_INDEX}"
print(f"Loaded {Ne} elements, {npe} nodes/element, nvar = {nvar}")
print(f"Plotting gradient of '{var_label}' (variable index {VAR_INDEX})")
print(f"  d({var_label})/dx range: [{bUx_flat.min():.4e}, {bUx_flat.max():.4e}]")
print(f"  d({var_label})/dy range: [{bUy_flat.min():.4e}, {bUy_flat.max():.4e}]")

# --- Build the figure ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(f"d({var_label})/dx", f"d({var_label})/dy"),
    horizontal_spacing=0.14,
)

# Use a symmetric diverging colorscale centred on zero,
# since gradients can be positive or negative.
def sym_range(a):
    m = np.max(np.abs(a))
    return (-m, m) if m > 0 else (-1.0, 1.0)

zmin_x, zmax_x = sym_range(bUx_flat)
zmin_y, zmax_y = sym_range(bUy_flat)

fig.add_trace(
    go.Scattergl(
        x=x_flat, y=y_flat,
        mode="markers",
        marker=dict(
            size=4,
            color=bUx_flat,
            colorscale="RdBu_r",
            cmin=zmin_x, cmax=zmax_x,
            colorbar=dict(title=f"d({var_label})/dx", x=0.45),
            showscale=True,
        ),
        hovertemplate=(
            "x = %{x:.4f}<br>"
            "y = %{y:.4f}<br>"
            f"d({var_label})/dx = "
            "%{marker.color:.4e}<extra></extra>"
        ),
        name=f"d{var_label}/dx",
    ),
    row=1, col=1,
)

fig.add_trace(
    go.Scattergl(
        x=x_flat, y=y_flat,
        mode="markers",
        marker=dict(
            size=4,
            color=bUy_flat,
            colorscale="RdBu_r",
            cmin=zmin_y, cmax=zmax_y,
            colorbar=dict(title=f"d({var_label})/dy", x=1.0),
            showscale=True,
        ),
        hovertemplate=(
            "x = %{x:.4f}<br>"
            "y = %{y:.4f}<br>"
            f"d({var_label})/dy = "
            "%{marker.color:.4e}<extra></extra>"
        ),
        name=f"d{var_label}/dy",
    ),
    row=1, col=2,
)

# Equal aspect ratio per panel
fig.update_yaxes(scaleanchor="x",  scaleratio=1, row=1, col=1)
fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)

fig.update_xaxes(title_text="x / l_0", row=1, col=1)
fig.update_xaxes(title_text="x / l_0", row=1, col=2)
fig.update_yaxes(title_text="y / l_0", row=1, col=1)
fig.update_yaxes(title_text="y / l_0", row=1, col=2)

fig.update_layout(
    title=f"Baseflow gradient of {var_label} ({Ne} elements, {npe} nodes/elem)",
    width=1500,
    height=750,
    showlegend=False,
    hovermode="closest",
)

fig.show()