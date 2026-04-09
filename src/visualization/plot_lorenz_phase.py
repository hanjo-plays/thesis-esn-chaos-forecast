# 3D phase portrait of the attractor
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
data = pd.read_csv(PROJECT_ROOT / "data" / "lorenz_raw.csv")
x = data['x'].values
y = data['y'].values
z = data['z'].values

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, color='teal', alpha=0.8, lw=0.5)

ax.set_xlabel('x', fontsize=12, labelpad=10)
ax.set_ylabel('y', fontsize=12, labelpad=10)
ax.set_zlabel('z', fontsize=12, labelpad=10)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

output_file = "lorenz_phase_space.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_file}")

plt.show()
