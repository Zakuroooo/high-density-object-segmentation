"""
metrics.py — one definition of every score in this project.

The headline fix this module exists for: the old results table published a
"Mean IoU" for Watershed and KMeans, but the baseline functions returned an
integer count and nothing else. There was no mask to compute IoU against, so
those two numbers could not have come from this repository. Here every method
returns masks, and every method is scored by the same function.
"""

from __future__ import annotations

import numpy as np

from config import COUNT_TOLERANCE


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean masks of identical shape."""
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


def match_instances(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
) -> tuple[list[tuple[int, int, float]], float]:
    """
    Greedy one-to-one matching between predicted and ground-truth instances.

    Class-agnostic on purpose. Watershed and KMeans cluster pixels and have no
    concept of an object category, so scoring them class-aware would compare
    them against a task they never attempt. To keep the four methods on one
    scale, every method is scored on instance overlap alone.

    Matching is greedy by descending IoU: repeatedly take the highest-IoU pair
    among still-unmatched instances. Greedy is not guaranteed optimal the way
    Hungarian assignment is, but on these mask counts the two agree in practice
    and greedy is the convention the detection literature uses for this report.

    Every ground-truth instance contributes to the mean. An unmatched one
    contributes 0.0 rather than being dropped — so a method that finds three
    objects out of twenty cannot score well by being right about the three.

    Returns:
        (matches, mean_iou) where matches is a list of (pred_idx, gt_idx, iou).
    """
    if not gt_masks:
        return [], 0.0
    if not pred_masks:
        return [], 0.0

    pairs = []
    for pi, pm in enumerate(pred_masks):
        for gi, gm in enumerate(gt_masks):
            iou = mask_iou(pm, gm)
            if iou > 0.0:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, iou))

    # Denominator is the ground-truth count: missed objects score zero.
    total = sum(m[2] for m in matches)
    return matches, total / len(gt_masks)


def count_metrics(pred_count: int, gt_count: int) -> dict:
    """Per-image counting scores."""
    err = abs(pred_count - gt_count)
    return {
        "pred_count": int(pred_count),
        "gt_count": int(gt_count),
        "abs_error": int(err),
        "within_tolerance": bool(err <= COUNT_TOLERANCE),
    }


def aggregate(per_image: list[dict]) -> dict:
    """
    Aggregate per-image records into the numbers that get published.

    Timing is reported as a median plus an interquartile range, not a mean. The
    old results reported a mean inference time that a single 12.7-second cold
    start had dragged from ~96 ms to 1650 ms — a mean over a warm-up outlier
    describes nothing.
    """
    n = len(per_image)
    if n == 0:
        return {}

    ious = [r["mean_iou"] for r in per_image]
    errs = [r["abs_error"] for r in per_image]
    hits = [r["within_tolerance"] for r in per_image]
    times = sorted(r["inference_ms"] for r in per_image)

    def pct(p: float) -> float:
        return float(times[min(int(p * len(times)), len(times) - 1)])

    return {
        "n_images": n,
        "mean_iou": round(float(np.mean(ious)), 4),
        "count_accuracy_pct": round(100.0 * sum(hits) / n, 1),
        "mae": round(float(np.mean(errs)), 2),
        "median_inference_ms": round(float(np.median(times)), 1),
        "iqr_inference_ms": [round(pct(0.25), 1), round(pct(0.75), 1)],
        "avg_pred_count": round(float(np.mean([r["pred_count"] for r in per_image])), 2),
        "avg_gt_count": round(float(np.mean([r["gt_count"] for r in per_image])), 2),
    }
