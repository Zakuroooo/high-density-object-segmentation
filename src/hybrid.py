"""
hybrid.py — the density-aware hybrid system.

This lived only inside a Colab notebook that was never executed and committed
with zero outputs, which meant the published Phase-3 numbers had no runnable
source. It is now ordinary importable code that the evaluation harness calls
like any other method.

The idea in one sentence: a detector tuned for ordinary scenes suppresses real
objects in crowded ones, so detect the crowding first and only then relax the
detector and cross-check the count against a classical method.

An honest note on what this method does and does not do, because the result
table shows it and it looks odd otherwise: the hybrid changes the COUNT estimate
much more than it changes mask quality. Its mask set comes from a YOLO pass, so
its IoU tracks YOLO's closely. Counting is the thing being improved.
"""

from __future__ import annotations

import cv2
import numpy as np

from baseline import watershed_segmentation
from config import (
    CONF,
    DENSE_CONF,
    DENSE_COUNT_THRESHOLD,
    DENSE_IOU_NMS,
    EDGE_DENSITY_THRESHOLD,
    IOU_NMS,
    WATERSHED_TRUST_RATIO,
    WATERSHED_WEIGHT,
    YOLO_WEIGHT,
)


def edge_density(image: np.ndarray) -> float:
    """
    Fraction of pixels that are Canny edges.

    A cheap, model-free proxy for visual clutter. Chosen over "just use the
    detector's own count" deliberately: if the detector has already failed to
    see the crowd, its count cannot be the signal that tells us a crowd is
    there. Canny costs well under a millisecond, so the gate is close to free.

    Thresholds 50/150 are the conventional Canny defaults. They were not tuned
    on the test set.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges)) / edges.size


def yolo_masks(model, image: np.ndarray, conf: float, iou: float) -> list[np.ndarray]:
    """Run YOLO and return per-instance boolean masks at the image's resolution."""
    res = model.predict(image, conf=conf, iou=iou, verbose=False)[0]
    if res.masks is None:
        return []
    h, w = image.shape[:2]
    out = []
    for m in res.masks.data.cpu().numpy():
        # YOLO returns masks at the network's stride, not the input size.
        resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out.append(resized > 0.5)
    return out


def hybrid_predict(model, image: np.ndarray) -> dict:
    """
    Run the full hybrid pipeline on one image.

    Returns a dict with:
        masks        — the instance masks the method commits to
        count        — its final count estimate (may exceed len(masks); see below)
        dense        — whether the density gate fired
        blended      — whether the watershed cross-check altered the count
        edge_density — the gate's input, kept for the ablation figure

    `count` and `len(masks)` can differ. When watershed finds substantially more
    regions than YOLO did, the method raises its count estimate without being
    able to say where those extra objects are. Reporting them as one number
    would imply a localisation the method never produced, so both are kept.
    """
    masks = yolo_masks(model, image, CONF, IOU_NMS)
    base_count = len(masks)

    ed = edge_density(image)
    dense = (ed > EDGE_DENSITY_THRESHOLD) or (base_count >= DENSE_COUNT_THRESHOLD)

    if dense:
        # Relaxed pass: lower confidence floor recovers instances that a crowded
        # scene pushes below threshold; looser NMS stops overlapping neighbours
        # in a crowd from suppressing one another.
        relaxed = yolo_masks(model, image, DENSE_CONF, DENSE_IOU_NMS)
        if len(relaxed) > len(masks):
            masks = relaxed

    count = len(masks)
    blended = False

    if dense:
        ws_count = len(watershed_segmentation(image))
        if count > 0 and ws_count > WATERSHED_TRUST_RATIO * count:
            count = int(round(YOLO_WEIGHT * count + WATERSHED_WEIGHT * ws_count))
            blended = True

    return {
        "masks": masks,
        "count": count,
        "dense": dense,
        "blended": blended,
        "edge_density": round(ed, 4),
    }
