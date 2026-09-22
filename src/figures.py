"""
figures.py — regenerate every published figure from the single results file.

Nothing here is hand-assembled. If `results/metrics/unified_results.json`
changes, every figure changes with it; if that file is missing, this script
refuses to draw rather than reusing a stale image. The previous version of the
project had twenty figures in three duplicated directories with no traceable
link back to the run that produced them.

Run from the project root:
    .venv/bin/python src/figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COUNT_TOLERANCE, FIGURES_DIR, METRICS_DIR

# ── Design tokens ─────────────────────────────────────────────────────────────
# Validated categorical palette (adjacent pairlist, light surface): all checks
# pass. The contrast check WARNs for aqua and yellow against this surface, which
# obligates visible direct labels — every bar in this file carries its value.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

SERIES = {
    "watershed": "#2a78d6",   # slot 1 blue
    "kmeans": "#eb6834",      # slot 2 orange
    "yolo": "#1baf7a",        # slot 3 aqua
    "hybrid": "#eda100",      # slot 4 yellow
}
LABEL = {
    "watershed": "Watershed",
    "kmeans": "KMeans",
    "yolo": "YOLOv8s-seg",
    "hybrid": "Hybrid",
}
ORDER = ["watershed", "kmeans", "yolo", "hybrid"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})


def _bars(ax, values, title, fmt="{:.3f}", higher_better=True):
    """One metric, four methods, direct-labelled. Never two scales on one axis."""
    xs = np.arange(len(ORDER))
    cols = [SERIES[k] for k in ORDER]
    bars = ax.bar(xs, values, color=cols, width=0.62,
                  edgecolor=SURFACE, linewidth=2)   # 2px surface gap between fills
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[k] for k in ORDER], rotation=18, ha="right")
    ax.set_title(title, color=INK, pad=10)
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    span = max(values) if max(values) else 1.0
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + span * 0.03, fmt.format(v),
                ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_ylim(0, span * 1.22)
    arrow = "higher is better" if higher_better else "lower is better"
    ax.set_xlabel(arrow, color=INK_2, fontsize=8.5, labelpad=8)


def fig_comparison(summary: dict, out: Path) -> None:
    """
    Three metrics, three panels — deliberately not one chart with two y-axes.

    IoU (0-1), accuracy (%) and MAE (objects) share no scale. A dual-axis version
    of this figure would let the reader infer crossings that do not exist.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    _bars(axes[0], [summary[k]["mean_iou"] for k in ORDER],
          "Mean instance IoU", "{:.3f}", True)
    _bars(axes[1], [summary[k]["count_accuracy_pct"] for k in ORDER],
          f"Count accuracy (within ±{COUNT_TOLERANCE})", "{:.0f}%", True)
    _bars(axes[2], [summary[k]["mae"] for k in ORDER],
          "Count MAE", "{:.2f}", False)
    fig.suptitle("Dense-scene segmentation: classical vs deep vs hybrid",
                 color=INK, fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_count_scatter(per_image: dict, out: Path) -> None:
    """
    Predicted vs true count, one panel per method — small multiples rather than
    four overlaid series, because an all-pairs scatter cannot clear the
    colour-separation floor at four categories.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.4))
    for ax, key in zip(axes.ravel(), ORDER):
        recs = per_image[key]
        gt = [r["gt_count"] for r in recs]
        pr = [r["pred_count"] for r in recs]
        hi = max(max(gt), max(pr)) * 1.08
        ax.plot([0, hi], [0, hi], color=INK_2, lw=1.2, ls="--", zorder=1)
        ax.fill_between([0, hi],
                        [-COUNT_TOLERANCE, hi - COUNT_TOLERANCE],
                        [COUNT_TOLERANCE, hi + COUNT_TOLERANCE],
                        color=INK_2, alpha=0.08, zorder=0)
        ax.scatter(gt, pr, s=42, color=SERIES[key], alpha=0.85,
                   edgecolor=SURFACE, linewidth=1.4, zorder=2)  # 2px surface ring
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_title(LABEL[key], color=INK)
        ax.set_xlabel("true object count")
        ax.set_ylabel("predicted count")
        ax.grid(alpha=0.7)
        ax.set_axisbelow(True)
    fig.suptitle(f"Predicted vs true count  ·  shaded band = ±{COUNT_TOLERANCE} tolerance",
                 color=INK, fontsize=12.5, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_inference(summary: dict, out: Path) -> None:
    """Median latency with the interquartile range, on a log scale."""
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ys = np.arange(len(ORDER))
    med = [summary[k]["median_inference_ms"] for k in ORDER]
    lo = [max(summary[k]["iqr_inference_ms"][0], 0.01) for k in ORDER]
    hi = [summary[k]["iqr_inference_ms"][1] for k in ORDER]
    err = [np.array(med) - np.array(lo), np.array(hi) - np.array(med)]
    ax.barh(ys, med, color=[SERIES[k] for k in ORDER], height=0.6,
            edgecolor=SURFACE, linewidth=2,
            xerr=err, error_kw={"ecolor": INK_2, "lw": 1.2, "capsize": 4})
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[k] for k in ORDER])
    ax.set_xscale("log")
    ax.set_xlabel("median inference time per image (ms, log scale) — lower is better")
    ax.grid(axis="x", alpha=0.7)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    for y, v in zip(ys, med):
        ax.text(v * 1.12, y, f"{v:.0f} ms", va="center", color=INK, fontsize=9)
    ax.set_title("Latency (bars show the interquartile range)", color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_density_gate(per_image: dict, out: Path) -> None:
    """
    What the hybrid's gate actually did: where it fired, and whether firing
    reduced the counting error relative to YOLO alone on the same image.
    """
    hyb = {r["img_id"]: r for r in per_image["hybrid"]}
    yolo = {r["img_id"]: r for r in per_image["yolo"]}
    ids = sorted(hyb)

    ed = [hyb[i]["edge_density"] for i in ids]
    delta = [yolo[i]["abs_error"] - hyb[i]["abs_error"] for i in ids]
    fired = [hyb[i]["dense"] for i in ids]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.axhline(0, color=INK_2, lw=1.2, ls="--")
    for state, colour, name in ((True, SERIES["hybrid"], "gate fired"),
                                (False, INK_2, "gate did not fire")):
        xs = [e for e, f in zip(ed, fired) if f is state]
        ys = [d for d, f in zip(delta, fired) if f is state]
        ax.scatter(xs, ys, s=46, color=colour, alpha=0.85,
                   edgecolor=SURFACE, linewidth=1.4, label=f"{name} (n={len(xs)})")
    ax.set_xlabel("Canny edge density")
    ax.set_ylabel("count error reduced vs YOLO alone\n(objects; positive = hybrid better)")
    ax.set_title("Where the density gate fires, and whether it helps", color=INK, pad=10)
    ax.grid(alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    res_path = METRICS_DIR / "unified_results.json"
    if not res_path.exists():
        sys.exit(f"Missing {res_path}. Run src/evaluate.py first — figures are "
                 f"never drawn from anything else.")

    data = json.loads(res_path.read_text())
    summary, per_image = data["summary"], data["per_image"]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_comparison(summary, FIGURES_DIR / "comparison.png")
    fig_count_scatter(per_image, FIGURES_DIR / "count_scatter.png")
    fig_inference(summary, FIGURES_DIR / "inference_time.png")
    fig_density_gate(per_image, FIGURES_DIR / "density_gate.png")

    prov = data["provenance"]
    print(f"Figures regenerated from {res_path.name} "
          f"(v{prov['experiment_version']}, {prov['date']}, n={prov['n_test_images']})")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {f.relative_to(FIGURES_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
