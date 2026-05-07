import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 1. Define absolute paths based on where this script lives
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / 'output' / 'tests'


def load_grid(prefix):
    """Load X/Y CSV pair written by the C++ test as `<prefix>_x.csv` etc."""
    x_file = OUTPUT_DIR / f"{prefix}_x.csv"
    y_file = OUTPUT_DIR / f"{prefix}_y.csv"
    try:
        X = pd.read_csv(x_file, header=None).values
        Y = pd.read_csv(y_file, header=None).values
        return X, Y
    except FileNotFoundError:
        return None, None


def plot_mesh(ax, X, Y, title):
    rows, cols = X.shape
    for i in range(rows):
        ax.plot(X[i, :], Y[i, :], 'k-', linewidth=0.5, alpha=0.6)
    for j in range(cols):
        ax.plot(X[:, j], Y[:, j], 'k-', linewidth=0.5, alpha=0.6)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')


# Each entry: (before_prefix, after_prefix, before_title, after_title, row_label)
panels = [
    ("testA_ortho_noisy",   "testA_ortho_smooth", "Noisy input",         "Smoothed",        "Test A: Pure smoothing"),
    ("testB_ortho_initial", "testB_ortho_final",  "Initial (algebraic)", "Orthogonalized",  "Test B: Curved bottom wall"),
]

# Filter out any panels whose files aren't present so the script still works
# if you've only run one test or have older outputs hanging around.
loaded = []
for before, after, btitle, atitle, label in panels:
    Xb, Yb = load_grid(before)
    Xa, Ya = load_grid(after)
    if Xb is None or Xa is None:
        print(f"[skip] {label}: missing CSVs for '{before}' or '{after}' in {OUTPUT_DIR}")
        continue
    loaded.append((Xb, Yb, Xa, Ya, btitle, atitle, label))

if not loaded:
    print(f"Error: No grid CSVs found in {OUTPUT_DIR}. Run the C++ test first.")
    raise SystemExit(1)

n_rows = len(loaded)
fig, axes = plt.subplots(n_rows, 2, figsize=(12, 6 * n_rows), squeeze=False)

for row_idx, (Xb, Yb, Xa, Ya, btitle, atitle, label) in enumerate(loaded):
    plot_mesh(axes[row_idx, 0], Xb, Yb, btitle)
    plot_mesh(axes[row_idx, 1], Xa, Ya, atitle)
    axes[row_idx, 0].text(
        -0.05, 0.5, label,
        transform=axes[row_idx, 0].transAxes,
        rotation=90, va='center', ha='right',
        fontsize=11, fontweight='bold',
    )

plt.tight_layout()
plt.show()