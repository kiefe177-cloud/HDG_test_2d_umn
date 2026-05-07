import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

# 1. Define absolute paths based on where this script lives
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / 'output' / 'tests'

# 2. Define target files in the output directory
spline_file = OUTPUT_DIR / "cpp_spline_output.csv"
knots_file = OUTPUT_DIR / "knots_data.csv"

# 3. Load Data
try:
    spline_data = pd.read_csv(spline_file, header=None)
    x_spline    = spline_data[0].values
    y_spline    = spline_data[1].values

    knots_data  = pd.read_csv(knots_file, header=None)
    x_knots     = knots_data[0].values
    y_knots     = knots_data[1].values

except FileNotFoundError:
    print(f"Error: CSV files not found in {OUTPUT_DIR}. Run the C++ test first.")
    exit()

# 4. Create Plot
fig = go.Figure()

# Trace 1: Smooth Spline Curve
fig.add_trace(go.Scatter(
    x=x_spline,
    y=y_spline,
    mode='lines',
    name='C++ Cubic Spline',
    line=dict(color='blue', width=4),
    hovertemplate='<b>Spline</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<extra></extra>'
))

# Trace 2: Original Knots
fig.add_trace(go.Scatter(
    x=x_knots,
    y=y_knots,
    mode='markers',
    name='Original Knots',
    marker=dict(
        size=12,
        color='red',
        symbol='circle',
        line=dict(width=2, color='black')
    ),
    hovertemplate='<b>Knot Point</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<extra></extra>'
))


# 5. Update Layout
fig.update_layout(
    title='C++ Spline Verification (Compare with MATLAB)',
    xaxis_title='X',
    yaxis_title='Y (sin(x))',
    template='plotly_white',
    hovermode="x unified"
)

fig.show()