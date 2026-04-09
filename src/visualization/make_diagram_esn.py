# ESN diagram for the thesis — node positions are random but seeded
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

C_INPUT  = "#4FC3F7"
C_RESERV = "#CE93D8"
C_RES_BG = "#F3E5F5"
C_RES_ED = "#6A1B9A"
C_OUTPUT = "#FF8A65"
C_EDGE   = "#0D47A1"
C_TRAIN  = "#E53935"
C_FIXED  = "#424242"
TITLE_FS = 13
LABEL_FS = 10
ANNOT_FS = 9
NODE_R   = 0.18

def draw_esn():
    fig, ax = plt.subplots(figsize=(10, 5))
    np.random.seed(42)

    # Input layer
    inp_x = 0.5
    inp_ys = [1.8, 2.6, 3.4]
    for y in inp_ys:
        c = patches.Circle((inp_x, y), NODE_R,
                            fc=C_INPUT, ec=C_EDGE, lw=1.2, zorder=10)
        ax.add_patch(c)
    ax.text(inp_x, 0.3, "Input Layer", ha="center", va="top",
            fontsize=LABEL_FS)

    # Reservoir
    res_cx, res_cy, res_r = 4.5, 2.6, 1.9
    bg = patches.Circle((res_cx, res_cy), res_r,
                         fc=C_RES_BG, ec=C_RES_ED, ls="--", lw=1.6, zorder=1)
    ax.add_patch(bg)
    ax.text(res_cx, 0.3, "Reservoir", ha="center", va="top", fontsize=LABEL_FS)

    n_nodes = 35
    angles = np.random.uniform(0, 2 * np.pi, n_nodes)
    radii  = np.random.uniform(0, res_r * 0.78, n_nodes)
    nx = res_cx + radii * np.cos(angles)
    ny = res_cy + radii * np.sin(angles)
    ax.scatter(nx, ny, s=22, color=C_RESERV, edgecolors=C_RES_ED,
               linewidths=0.4, zorder=5)

    # Sparse recurrent connections
    for i in range(n_nodes):
        targets = np.random.choice(n_nodes, 2, replace=False)
        for t in targets:
            ax.plot([nx[i], nx[t]], [ny[i], ny[t]],
                    color=C_RES_ED, lw=0.9, alpha=0.30, zorder=2)

    # Readout layer
    out_x = 8.5
    out_ys = [1.8, 2.6, 3.4]
    for y in out_ys:
        c = patches.Circle((out_x, y), NODE_R,
                            fc=C_OUTPUT, ec=C_EDGE, lw=1.2, zorder=10)
        ax.add_patch(c)
    ax.text(out_x, 0.3, "Readout Layer", ha="center", va="top",
            fontsize=LABEL_FS)

    # Input -> reservoir (fixed weights, dashed)
    for iy in inp_ys:
        tgts = np.random.choice(n_nodes, 4, replace=False)
        for t in tgts:
            ax.plot([inp_x + NODE_R, nx[t]], [iy, ny[t]],
                    color=C_FIXED, ls="--", lw=1.2, alpha=0.55, zorder=3)

    # Reservoir -> output (trainable weights, solid red)
    for i in range(n_nodes):
        for oy in out_ys:
            ax.plot([nx[i], out_x - NODE_R], [ny[i], oy],
                    color=C_TRAIN, lw=0.9, alpha=0.15, zorder=3)

    legend_handles = [
        Line2D([0], [0], color=C_FIXED, ls="--", lw=1.2,
               label="Fixed weights"),
        Line2D([0], [0], color=C_TRAIN, ls="-",  lw=1.2,
               label="Trainable weights"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              fontsize=ANNOT_FS, ncol=2, frameon=False,
              bbox_to_anchor=(0.5, -0.02))

    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.2, 5.2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "diagram_esn.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)

if __name__ == "__main__":
    draw_esn()
