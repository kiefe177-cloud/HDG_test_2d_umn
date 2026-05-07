import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# 1. Define absolute paths based on where this script lives
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / 'output' / 'tests'

def plot_results():
    # 2. Define target files in the output directory
    fine_file = OUTPUT_DIR / "interp_surface.csv"
    orig_file = OUTPUT_DIR / "orig_grid_y.csv"

    try:
        # Load the Fine Interpolated Surface (50x50)
        Z_fine = pd.read_csv(fine_file, header=None).values
        
        # Load the Original Coarse Grid (5x5)
        Z_orig = pd.read_csv(orig_file, header=None).values
    except FileNotFoundError:
        print(f"Error: Could not find files in {OUTPUT_DIR}. Run C++ test first!")
        return

    # Create coordinate grids
    # Fine grid (0 to 1)
    x_fine = np.linspace(0, 1, Z_fine.shape[0])
    y_fine = np.linspace(0, 1, Z_fine.shape[1])
    X_F, Y_F = np.meshgrid(x_fine, y_fine, indexing='ij')

    # Coarse grid (0 to 1)
    x_coarse = np.linspace(0, 1, Z_orig.shape[0])
    y_coarse = np.linspace(0, 1, Z_orig.shape[1])
    X_C, Y_C = np.meshgrid(x_coarse, y_coarse, indexing='ij')

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Plot the Smooth Interpolated Surface
    surf = ax.plot_surface(X_F, Y_F, Z_fine, cmap='viridis', alpha=0.6, edgecolor='none')
    
    # 2. Plot the Original Data Points (Red Dots)
    ax.scatter(X_C, Y_C, Z_orig, c='red', s=50, label='Original Grid Nodes')

    ax.set_title("Bicubic Spline Verification")
    ax.set_xlabel("Phi")
    ax.set_ylabel("Psi")
    ax.set_zlabel("Physical Y (Height)")
    ax.legend()
    
    plt.show()

if __name__ == "__main__":
    plot_results()