# LSTM cell diagram, loosely based on Colah's blog post
# gate positions and wiring tweaked by hand until it looked right
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

C_INPUT  = "#4FC3F7"
C_HIDDEN = "#81C784"
C_OUTPUT = "#FF8A65"
C_EDGE   = "#0D47A1"
C_CONN   = "#9E9E9E"
C_GATE   = "#FFF9C4"
C_OP     = "#FFCC80"
C_CELL   = "#E8F5E9"
TITLE_FS = 16
LABEL_FS = 13
ANNOT_FS = 12
GATE_FS  = 12

def _arrow(ax, x0, y0, x1, y1, colour="black", lw=1.2, zorder=5):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=13,
        color=colour, lw=lw, zorder=zorder,
    )
    ax.add_patch(a)

def _gate_box(ax, cx, cy, label, size=0.35):
    s = size
    rect = patches.FancyBboxPatch(
        (cx - s / 2, cy - s / 2), s, s,
        boxstyle="round,pad=0.06",
        facecolor=C_GATE, edgecolor=C_EDGE, lw=1.0, zorder=8,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=GATE_FS, fontweight="bold", color=C_EDGE, zorder=9)

def _op_circle(ax, cx, cy, label, radius=0.18):
    c = patches.Circle(
        (cx, cy), radius,
        facecolor=C_OP, edgecolor=C_EDGE, lw=1.0, zorder=8,
    )
    ax.add_patch(c)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=GATE_FS, fontweight="bold", color=C_EDGE, zorder=9)

def _line(ax, x0, y0, x1, y1, colour="black", lw=1.2):
    ax.plot([x0, x1], [y0, y1], color=colour, lw=lw, solid_capstyle="round",
            zorder=4)

def draw_lstm():
    fig, ax = plt.subplots(figsize=(10, 5))

    cell_left, cell_right = 2.0, 8.0
    cell_bot,  cell_top   = 0.8, 4.2
    cell_w = cell_right - cell_left
    cell_h = cell_top - cell_bot
    y_cell_state = 3.8
    y_hidden     = 1.2
    y_gates      = 2.2

    # Cell background
    bg = patches.FancyBboxPatch(
        (cell_left, cell_bot), cell_w, cell_h,
        boxstyle="round,pad=0.15",
        facecolor=C_CELL, edgecolor=C_EDGE, lw=1.6, zorder=1,
    )
    ax.add_patch(bg)

    # Cell state line
    _arrow(ax, cell_left - 0.6, y_cell_state, cell_right + 0.6, y_cell_state,
           colour=C_EDGE, lw=1.2, zorder=3)
    ax.text((cell_left + cell_right) / 2, y_cell_state + 0.18,
            "Cell state  $C_t$", ha="center", va="bottom",
            fontsize=ANNOT_FS, fontstyle="italic", color=C_EDGE)

    # Gate positions
    g_spacing = cell_w / 5
    fg_x  = cell_left + g_spacing
    ig_x  = cell_left + 2 * g_spacing
    itg_x = cell_left + 2.6 * g_spacing
    og_x  = cell_left + 3.6 * g_spacing

    # Forget gate
    _gate_box(ax, fg_x, y_gates, "$\\sigma$")
    ax.text(fg_x, y_gates - 0.32, "forget ($f_t$)", ha="center", va="top",
            fontsize=ANNOT_FS - 1, color="#616161",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8),
            zorder=10)

    # Input gate
    _gate_box(ax, ig_x, y_gates, "$\\sigma$")
    ax.text(ig_x, y_gates - 0.32, "input ($i_t$)", ha="center", va="top",
            fontsize=ANNOT_FS - 1, color="#616161",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8),
            zorder=10)

    # Input candidate (tanh)
    _gate_box(ax, itg_x, y_gates, "tanh")
    ax.text(itg_x, y_gates - 0.32, "$\\tilde{C}_t$", ha="center", va="top",
            fontsize=ANNOT_FS - 1, color="#616161",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8),
            zorder=10)

    # Output gate
    _gate_box(ax, og_x, y_gates, "$\\sigma$")
    ax.text(og_x, y_gates - 0.32, "output ($o_t$)", ha="center", va="top",
            fontsize=ANNOT_FS - 1, color="#616161",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8),
            zorder=10)

    # Pointwise operations on cell state
    _op_circle(ax, fg_x, y_cell_state, "×")
    _op_circle(ax, (ig_x + itg_x) / 2, y_cell_state, "+")

    stack_x = cell_left + 4.2 * g_spacing
    tanh_out_x = stack_x
    tanh_out_y = y_cell_state - 0.55
    _gate_box(ax, tanh_out_x, tanh_out_y, "tanh", size=0.32)

    op_out_x = stack_x
    op_out_y = y_cell_state - 1.05
    _op_circle(ax, op_out_x, op_out_y, "×")

    # Gate -> cell state wiring
    _line(ax, fg_x, y_gates + 0.175, fg_x, y_cell_state - 0.18)

    mid_add_x = (ig_x + itg_x) / 2
    _line(ax, ig_x, y_gates + 0.175, ig_x, y_gates + 0.6)
    _line(ax, ig_x, y_gates + 0.6, mid_add_x - 0.25, y_gates + 0.6)
    _line(ax, itg_x, y_gates + 0.175, itg_x, y_gates + 0.6)
    _line(ax, itg_x, y_gates + 0.6, mid_add_x + 0.25, y_gates + 0.6)

    _op_circle(ax, mid_add_x, y_gates + 0.6, "×", radius=0.15)
    _line(ax, mid_add_x, y_gates + 0.75, mid_add_x, y_cell_state - 0.18)

    # Cell state -> tanh -> × -> h_t
    _line(ax, tanh_out_x, y_cell_state, tanh_out_x, tanh_out_y + 0.16)
    _line(ax, tanh_out_x, tanh_out_y - 0.16, op_out_x, op_out_y + 0.18)

    # Output gate -> × (route right from σ to ×)
    _line(ax, og_x, y_gates + 0.175, og_x, op_out_y)
    _line(ax, og_x, op_out_y, op_out_x - 0.18, op_out_y)

    # h_{t-1} input from left
    _arrow(ax, cell_left - 0.6, y_hidden, cell_left + 0.3, y_hidden,
           colour=C_EDGE, lw=1.2)
    ax.text(cell_left - 0.75, y_hidden, "$h_{t-1}$",
            ha="right", va="center", fontsize=LABEL_FS, color=C_EDGE)

    # h_{t-1} fans out to all gates
    branch_x = cell_left + 0.5
    _line(ax, branch_x, y_hidden, branch_x, y_gates)
    for gx in [fg_x, ig_x, itg_x, og_x]:
        _line(ax, branch_x, y_gates, gx - 0.175, y_gates)

    # h_t output to right
    _line(ax, op_out_x, op_out_y - 0.18, op_out_x, y_hidden)
    _arrow(ax, op_out_x, y_hidden, cell_right + 0.6, y_hidden,
           colour=C_EDGE, lw=1.2)
    ax.text(cell_right + 0.75, y_hidden, "$h_t$",
            ha="left", va="center", fontsize=LABEL_FS, color=C_EDGE)

    # x_t input from below
    xt_x = (fg_x + og_x) / 2
    _arrow(ax, xt_x, cell_bot - 0.6, xt_x, y_hidden,
           colour="black", lw=1.2)
    ax.text(xt_x, cell_bot - 0.75, "$x_t$",
            ha="center", va="top", fontsize=LABEL_FS)

    # h_t upward output
    ht_up_x = cell_right + 0.0
    _arrow(ax, ht_up_x, y_hidden + 0.05, ht_up_x, cell_top + 0.4,
           colour=C_EDGE, lw=1.2)
    ax.text(ht_up_x + 0.15, cell_top + 0.5, "$h_t$",
            ha="left", va="bottom", fontsize=LABEL_FS, color=C_EDGE)

    # Neighbour cells
    for nx, label in [(0.4, "$A_{t-1}$"), (cell_right + 1.6, "$A_{t+1}$")]:
        nw, nh = 1.0, cell_h * 0.6
        ny = (cell_bot + cell_top) / 2 - nh / 2
        rect = patches.FancyBboxPatch(
            (nx - nw / 2, ny), nw, nh,
            boxstyle="round,pad=0.10",
            facecolor=C_HIDDEN, edgecolor=C_EDGE, lw=1.2, alpha=0.45,
            zorder=1,
        )
        ax.add_patch(rect)
        ax.text(nx, ny + nh / 2, label, ha="center", va="center",
                fontsize=LABEL_FS, color=C_EDGE, alpha=0.7)

    # Cell state arrows to neighbours
    _arrow(ax, 0.9, y_cell_state, cell_left - 0.6, y_cell_state,
           colour=C_EDGE, lw=1.2)
    _arrow(ax, cell_right + 0.6, y_cell_state,
           cell_right + 1.1, y_cell_state, colour=C_EDGE, lw=1.2)

    ax.text(cell_left - 0.8, y_cell_state + 0.18, "$C_{t-1}$",
            ha="center", va="bottom", fontsize=ANNOT_FS, color=C_EDGE,
            zorder=10)
    ax.text(cell_right + 0.3, y_cell_state + 0.18, "$C_t$",
            ha="center", va="bottom", fontsize=ANNOT_FS, color=C_EDGE)

    ax.set_xlim(-0.8, cell_right + 2.6)
    ax.set_ylim(-0.2, cell_top + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "diagram_lstm.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)

if __name__ == "__main__":
    draw_lstm()
