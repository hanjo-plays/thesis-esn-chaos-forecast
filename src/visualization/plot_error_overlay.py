# All error curves on one plot — linear and log scale

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
FIGURES_DIR  = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("ESN  W=10",  "esn_w10_error_curve.npy",  "#1b9e77", "-"),
    ("ESN  W=1",   "esn_w1_error_curve.npy",   "#66c2a5", "--"),
    ("FFNN W=10",  "ffnn_w10_error_curve.npy",  "#d95f02", "-"),
    ("FFNN W=1",   "ffnn_w1_error_curve.npy",   "#fc8d62", "--"),
    ("LSTM W=1",   "lstm_w1_error_curve.npy",   "#7570b3", "--"),
    ("LSTM W=10",  "lstm_w10_error_curve.npy",  "#e7298a", "-"),
]

curves = {}
for label, fname, _, _ in MODELS:
    path = MODELS_DIR / fname
    if not path.exists():
        print(f"WARNING: {path} not found — skipping {label}")
        continue
    curves[label] = np.load(path)

steps = np.arange(1, 201)

fig, (ax_lin, ax_log) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

for label, fname, colour, linestyle in MODELS:
    if label not in curves:
        continue
    err = curves[label]
    ax_lin.plot(steps, err, color=colour, ls=linestyle, lw=1.4, label=label)
    ax_log.plot(steps, err, color=colour, ls=linestyle, lw=1.4, label=label)

ax_lin.axhline(2.0, color="grey", ls=":", lw=0.8, label="ε = 2 threshold")
ax_lin.set_xlabel("Forecast step", fontsize=14)
ax_lin.set_ylabel("Euclidean error", fontsize=14)
ax_lin.legend(fontsize=11, loc="upper left")
ax_lin.tick_params(labelsize=12)
ax_lin.set_xlim(1, 200)

ax_log.axhline(2.0, color="grey", ls=":", lw=0.8, label="ε = 2 threshold")
ax_log.set_yscale("log")
ax_log.set_xlabel("Forecast step", fontsize=14)
ax_log.set_ylabel("Euclidean error (log)", fontsize=14)
ax_log.legend(fontsize=11, loc="lower right")
ax_log.tick_params(labelsize=12)
ax_log.set_xlim(1, 200)

out_path = FIGURES_DIR / "error_curve_overlay.png"
fig.savefig(out_path, dpi=300)
plt.close(fig)
print(f"Saved: {out_path}")
