# FFNN architecture diagram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

C_INPUT  = "#4FC3F7"
C_HIDDEN = "#81C784"
C_OUTPUT = "#FF8A65"
C_EDGE   = "#0D47A1"
C_CONN   = "#9E9E9E"
TITLE_FS = 13
LABEL_FS = 10
ANNOT_FS = 9
NODE_R   = 0.22

def draw_ffnn():
    fig, ax = plt.subplots(figsize=(10, 5))

    layers = [
        (6, "Input Layer",      C_INPUT),
        (8, "Hidden Layer 1",   C_HIDDEN),
        (8, "Hidden Layer 2",   C_HIDDEN),
        (3, "Output Layer",     C_OUTPUT),
    ]

    h_gap = 2.8
    v_gap = 0.80
    max_n = max(n for n, *_ in layers)

    coords = {}

    for li, (n, label, colour) in enumerate(layers):
        x = li * h_gap
        y_off = (max_n - n) * v_gap / 2
        coords[li] = []
        for ni in range(n):
            y = y_off + ni * v_gap
            coords[li].append((x, y))
            circle = patches.Circle(
                (x, y), NODE_R,
                facecolor=colour, edgecolor=C_EDGE,
                lw=1.2, zorder=10,
            )
            ax.add_patch(circle)
        ax.text(x, -0.7, label, ha="center", va="top",
                fontsize=LABEL_FS, linespacing=1.3)

    # Fully connected lines between consecutive layers
    for li in range(len(layers) - 1):
        for sx, sy in coords[li]:
            for ex, ey in coords[li + 1]:
                ax.plot([sx, ex], [sy, ey],
                        color=C_CONN, alpha=0.35, lw=1.2, zorder=1)

    ax.set_aspect("equal")
    ax.set_xlim(-1.0, (len(layers) - 1) * h_gap + 1.0)
    ax.set_ylim(-1.6, max_n * v_gap - 0.2)
    ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "diagram_ffnn.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)

if __name__ == "__main__":
    draw_ffnn()
