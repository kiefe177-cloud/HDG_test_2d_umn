import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent / 'output'/ 'cyl_driver' / 'amr_leaves.csv'

leaves = pd.read_csv(CSV_PATH, header=None).values

fig, ax = plt.subplots(figsize=(8, 8))
for x0, y0, x2, y2 in leaves:
    rect = patches.Rectangle(
        (x0, y0), x2 - x0, y2 - y0,
        linewidth=0.6, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

ax.set_xlim(leaves[:, 0].min(), leaves[:, 2].max())
ax.set_ylim(leaves[:, 1].min(), leaves[:, 3].max())
ax.set_aspect('equal')
ax.set_title(f'AMR mesh ({len(leaves)} leaves)')
plt.tight_layout()
plt.show()