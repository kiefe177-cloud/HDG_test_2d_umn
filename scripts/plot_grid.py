import plotly.graph_objects as go
import pandas as pd
import numpy as np

# 1. Load Data
try:
    X = pd.read_csv("grid_x.csv", header=None).values
    Y = pd.read_csv("grid_y.csv", header=None).values
    Z = pd.read_csv("grid_z.csv", header=None).values
except FileNotFoundError:
    print("Error: CSV files not found. Are you running this in the 'build' folder?")
    exit()

# 2. Generate Custom Labels (i, j)
rows, cols = X.shape
hover_labels = []

# Loop through every point to create a label string
for i in range(rows):
    for j in range(cols):
        # <br> is HTML for a line break
        label = f"Index: ({i}, {j})<br>X: {X[i,j]:.2f}<br>Y: {Y[i,j]:.2f}<br>Z: {Z[i,j]:.2f}"
        hover_labels.append(label)

# 3. Create Interactive Scatter Plot
fig = go.Figure(data=[go.Scatter3d(
    x=X.flatten(),  # Flatten matrices to 1D lists for scatter plots
    y=Y.flatten(),
    z=Z.flatten(),
    mode='markers', # Use 'lines+markers' if you want to see grid lines too
    marker=dict(
        size=5,
        color=Z.flatten(),      # Color dots based on height (Z)
        colorscale='Viridis',
        opacity=0.8
    ),
    text=hover_labels,  # Attach our custom list of labels
    hoverinfo='text'    # Tell Plotly to use ONLY our custom text
)])

fig.update_layout(
    title='Interactive Grid with (i, j) Labels',
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Z Axis'
    )
)

fig.show()