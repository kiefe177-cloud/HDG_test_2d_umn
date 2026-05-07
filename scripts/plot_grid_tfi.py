import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Define absolute paths based on where this script lives
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / 'output' / 'tests'

def plot_grid():
    # 2. Define target files in the output directory
    x_file = OUTPUT_DIR / "grid_x_tfi.csv"
    y_file = OUTPUT_DIR / "grid_y_tfi.csv"

    # 3. Load Data
    try:
        if not x_file.exists():
            print(f"Error: Could not find {x_file}. Run the C++ test first.")
            return

        # Read CSVs (header=None assumes no title row)
        X = pd.read_csv(x_file, header=None).values
        Y = pd.read_csv(y_file, header=None).values
        
        # Optional: Read Z if you want to plot 3D later, but we focus on 2D mesh here
        # z_file = OUTPUT_DIR / "grid_z_tfi.csv"
        # Z = pd.read_csv(z_file, header=None).values

    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    print(f"Loaded Grid: {X.shape}")

    # 4. Setup Plot
    plt.figure(figsize=(8, 8))
    plt.title(f"Grid Visualization ({X.shape[0]}x{X.shape[1]})")
    plt.xlabel("X")
    plt.ylabel("Y")

    # 5. Plot Grid Lines
    rows, cols = X.shape

    # Plot "Horizontal" lines (varying columns, fixed row)
    for i in range(rows):
        plt.plot(X[i, :], Y[i, :], 'k-', linewidth=1.0, alpha=0.6)

    # Plot "Vertical" lines (varying rows, fixed column)
    for j in range(cols):
        plt.plot(X[:, j], Y[:, j], 'k-', linewidth=1.0, alpha=0.6)

    # 6. Highlight Corners/Boundaries
    plt.scatter(X, Y, c='red', s=10, zorder=5, label='Nodes') # Nodes
    
    # Highlight specific corners
    plt.plot(X[0,0], Y[0,0], 'bo', markersize=8, label='Origin (0,0)')
    plt.plot(X[-1,-1], Y[-1,-1], 'go', markersize=8, label='Max Corner')

    plt.axis('equal') # Important to see true geometry
    plt.grid(False)   # Turn off background grid since we are plotting the mesh
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_grid()