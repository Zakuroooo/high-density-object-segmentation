"""
ablation.py — take the hybrid apart and find out which half is doing the work.

The hybrid has two mechanisms, and the headline number does not say how much
each contributes:

  1. a density gate that triggers a second, relaxed YOLO pass, and
  2. a watershed cross-check that can raise the final count estimate.

Reporting only the combined result would let a reader assume both matter. They
do not, and this script is what establishes that. It is deliberately a separate,
runnable analysis rather than a paragraph in the README, because a claim about
one's own method should be reproducible by the person reading it.

Reads `results/metrics/unified_results.json`; writes `results/metrics/ablation.json`
and one figure. Run after `src/evaluate.py`:

    PYTHONPATH=src .venv/bin/python src/ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COUNT_TOLERANCE, FIGURES_DIR, METRICS_DIR


def subset_stats(ids, yolo, hybrid) -> dict:
    """Counting performance of YOLO vs the hybrid over a subset of images."""
    if not ids:
        return {"n": 0}
    n = len(ids)
    return {
        "n": n,
        "yolo_mae": round(sum(yolo[i]["abs_error"] for i in ids) / n, 2),
        "hybrid_mae": round(sum(hybrid[i]["abs_error"] for i in ids) / n, 2),
        "yolo_within_tol": sum(yolo[i]["within_tolerance"] for i in ids),
        "hybrid_within_tol": sum(hybrid[i]["within_tolerance"] for i in ids),
        "objects_recovered": sum(
            yolo[i]["abs_error"] - hybrid[i]["abs_error"] for i in ids),
    }


def main() -> None:
    res_path = METRICS_DIR / "unified_results.json"
    if not res_path.exists():
        sys.exit(f"Missing {res_path}. Run src/evaluate.py first.")

    data = json.loads(res_path.read_text())
    per = data["per_image"]
    yolo = {r["img_id"]: r for r in per["yolo"]}
    hybrid = {r["img_id"]: r for r in per["hybrid"]}
    ids = sorted(yolo)

    gate_fired = [i for i in ids if hybrid[i]["dense"]]
    gate_quiet = [i for i in ids if not hybrid[i]["dense"]]
    blended = [i for i in ids if hybrid[i]["blended"]]

    overall = subset_stats(ids, yolo, hybrid)
    from_blend = sum(
        yolo[i]["abs_error"] - hybrid[i]["abs_error"] for i in blended)
    total = overall["objects_recovered"]

    out = {
        "question": "Which of the hybrid's two mechanisms produces its gain?",
        "count_tolerance": COUNT_TOLERANCE,
        "overall": overall,
        "gate_fired": subset_stats(gate_fired, yolo, hybrid),
        "gate_did_not_fire": subset_stats(gate_quiet, yolo, hybrid),
        "watershed_blend_applied": subset_stats(blended, yolo, hybrid),
        "attribution": {
            "total_objects_recovered": total,
            "from_watershed_cross_check": from_blend,
            "from_relaxed_detector_pass": total - from_blend,
            "watershed_share_pct": round(100 * from_blend / total, 1) if total else 0.0,
        },
        "per_blended_image": [
            {
                "img_id": i,
                "gt": yolo[i]["gt_count"],
                "yolo": yolo[i]["pred_count"],
                "hybrid": hybrid[i]["pred_count"],
                "error_before": yolo[i]["abs_error"],
                "error_after": hybrid[i]["abs_error"],
            }
            for i in blended
        ],
        "conclusion": (
            "The density-gated relaxed detector pass accounts for "
            f"{total - from_blend} of the {total} objects of count-error "
            f"reduction; the watershed cross-check accounts for {from_blend}, "
            f"firing on only {len(blended)} of {len(ids)} images. The method is "
            "therefore best described as density-aware two-pass detection with a "
            "marginal classical correction — not as an equal partnership between "
            "a deep and a classical method. On the "
            f"{len(gate_quiet)} images where the gate did not fire, the hybrid is "
            "identical to YOLO, which is the intended behaviour: the gate costs "
            "nothing when it is not needed."
        ),
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "ablation.json").write_text(json.dumps(out, indent=2))

    # ── Figure: attribution of the gain ───────────────────────────────────────
    SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
        "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "font.size": 10,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # Single stacked bar: where the improvement comes from.
    a = out["attribution"]
    ax1.barh([0], [a["from_relaxed_detector_pass"]], color="#eda100",
             height=0.5, edgecolor=SURFACE, linewidth=2,
             label=f"relaxed detector pass ({a['from_relaxed_detector_pass']})")
    ax1.barh([0], [a["from_watershed_cross_check"]],
             left=[a["from_relaxed_detector_pass"]], color="#2a78d6",
             height=0.5, edgecolor=SURFACE, linewidth=2,
             label=f"watershed cross-check ({a['from_watershed_cross_check']})")
    ax1.set_yticks([])
    ax1.set_xlabel("objects of count error removed vs YOLO alone")
    ax1.set_title(f"Where the hybrid's gain comes from\n"
                  f"watershed contributes {a['watershed_share_pct']}%",
                  color=INK, pad=10)
    ax1.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=1)
    ax1.grid(axis="x", alpha=0.7)
    ax1.set_axisbelow(True)

    # MAE before/after, split by whether the gate fired.
    groups = [("gate fired", out["gate_fired"]),
              ("gate quiet", out["gate_did_not_fire"])]
    xs = range(len(groups))
    w = 0.36
    ax2.bar([x - w / 2 for x in xs], [g[1]["yolo_mae"] for g in groups],
            width=w, color="#1baf7a", edgecolor=SURFACE, linewidth=2, label="YOLO alone")
    ax2.bar([x + w / 2 for x in xs], [g[1]["hybrid_mae"] for g in groups],
            width=w, color="#eda100", edgecolor=SURFACE, linewidth=2, label="Hybrid")
    for x, (_, g) in zip(xs, groups):
        ax2.text(x - w / 2, g["yolo_mae"] + 0.08, f"{g['yolo_mae']:.2f}",
                 ha="center", fontsize=9, color=INK)
        ax2.text(x + w / 2, g["hybrid_mae"] + 0.08, f"{g['hybrid_mae']:.2f}",
                 ha="center", fontsize=9, color=INK)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([f"{g[0]}\n(n={g[1]['n']})" for g in groups])
    ax2.set_ylabel("count MAE — lower is better")
    ax2.set_title("The gate does nothing when it does not fire", color=INK, pad=10)
    ax2.legend(frameon=False)
    ax2.grid(axis="y", alpha=0.7)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "ablation_attribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(out["conclusion"])
    print(f"\n  -> {METRICS_DIR / 'ablation.json'}")
    print(f"  -> {FIGURES_DIR / 'ablation_attribution.png'}")


if __name__ == "__main__":
    main()
