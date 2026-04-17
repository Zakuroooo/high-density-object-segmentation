"""
create_architecture_diagram.py  —  Generates the YOLOv8 architecture diagram
for results/figures/architecture_diagram.png.

Run from the project root:
    python src/create_architecture_diagram.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def draw_box(ax, x, y, w, h, text, color="#2C3E50", textcolor="white",
             fontsize=9, style="round,pad=0.1"):
    """Draw a rounded box with centred text on `ax`."""
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle=style,
                         facecolor=color, edgecolor="white",
                         linewidth=1.5, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            color=textcolor, fontsize=fontsize,
            fontweight="bold", zorder=4, wrap=True,
            multialignment="center")


def draw_arrow(ax, x1, y1, x2, y2, color="#95A5A6"):
    """Draw a downward arrow between two points."""
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, mutation_scale=18),
                zorder=2)


def main():
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_facecolor("#1A1A2E")
    fig.patch.set_facecolor("#1A1A2E")

    fig.suptitle("YOLOv8s-seg Architecture", fontsize=16,
                 fontweight="bold", color="white", y=0.97)

    # ── Layer definitions: (x_centre, y_centre, width, height, label, colour) ──
    cx = 5.5   # centre x for main column

    layers = [
        # x,   y,    w,    h,   label,                       colour
        (cx,   13.0, 4.5,  0.7, "Input Image  640×640×3",   "#1ABC9C"),
        (cx,   11.8, 4.5,  0.9, "Backbone\nCSPDarknet53",   "#2980B9"),
        (cx,   10.5, 4.8,  0.7, "Feature maps @ 3 scales\n(P3/8, P4/16, P5/32)", "#34495E"),
        (cx,    9.2, 4.5,  0.9, "Neck\nFPN + PAN",          "#8E44AD"),
        (cx,    7.9, 4.8,  0.7, "Multi-scale feature fusion","#34495E"),
    ]

    for (x, y, w, h, label, color) in layers:
        draw_box(ax, x, y, w, h, label, color=color, fontsize=9)

    # Arrow chain for main column
    arrow_ys = [(13.0 - 0.35, 11.8 + 0.45),
                (11.8 - 0.45, 10.5 + 0.35),
                (10.5 - 0.35, 9.2 + 0.45),
                ( 9.2 - 0.45, 7.9 + 0.35)]
    for (y1, y2) in arrow_ys:
        draw_arrow(ax, cx, y1, cx, y2)

    # ── Fork into Detection Head and Segmentation Head ──────────────────────
    arrow_y_start = 7.9 - 0.35

    # Left: detection head
    lx = 2.8
    rx = 8.2
    fork_y = 6.8

    draw_arrow(ax, cx, arrow_y_start, lx, fork_y + 0.45)
    draw_arrow(ax, cx, arrow_y_start, rx, fork_y + 0.45)

    draw_box(ax, lx, fork_y, 4.2, 0.8, "Detection Head\nBounding boxes + classes",
             color="#E74C3C", fontsize=8.5)
    draw_box(ax, rx, fork_y, 4.2, 0.8, "Segmentation Head\nInstance masks (32 proto)",
             color="#F39C12", fontsize=8.5)

    # Sub-steps below each head
    draw_arrow(ax, lx, fork_y - 0.4, lx, 5.6 + 0.35)
    draw_arrow(ax, rx, fork_y - 0.4, rx, 5.6 + 0.35)

    draw_box(ax, lx, 5.6, 4.2, 0.65, "NMS Post-processing",
             color="#C0392B", fontsize=8.5)
    draw_box(ax, rx, 5.6, 4.2, 0.65, "Mask Refinement\n(multiply proto × coeffs)",
             color="#D35400", fontsize=8.5)

    # Merge back
    draw_arrow(ax, lx, 5.6 - 0.325, cx, 4.3 + 0.35)
    draw_arrow(ax, rx, 5.6 - 0.325, cx, 4.3 + 0.35)

    draw_box(ax, cx, 4.3, 5.5, 0.8,
             "Final Output\nBoxes + Masks + Class Labels + Confidence",
             color="#27AE60", fontsize=8.5)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color="#1ABC9C", label="Input"),
        mpatches.Patch(color="#2980B9", label="Backbone"),
        mpatches.Patch(color="#8E44AD", label="Neck"),
        mpatches.Patch(color="#E74C3C", label="Detection Head"),
        mpatches.Patch(color="#F39C12", label="Segmentation Head"),
        mpatches.Patch(color="#27AE60", label="Output"),
    ]
    ax.legend(handles=legend_items, loc="lower center", ncol=3,
              framealpha=0.15, labelcolor="white", fontsize=8.5,
              facecolor="#1A1A2E", edgecolor="#555555",
              bbox_to_anchor=(0.5, 0.01))

    out_path = os.path.join("results", "figures", "architecture_diagram.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Architecture diagram saved → {out_path}")


if __name__ == "__main__":
    main()
