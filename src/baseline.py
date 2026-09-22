"""
baseline.py — classical segmentation baselines.

Both methods now return instance MASKS, not just a count. That is the substantive
change from the first version: the old functions returned an integer, so the
"Mean IoU" this project published for Watershed and KMeans had no mask to be
computed from. Returning masks makes those rows real and, as it turns out,
makes the baselines look worse than the numbers that were published for them.

Neither method has any notion of object class. They cluster pixels. That is the
honest limitation of the classical approach on natural images and it is the
reason the whole comparison is scored class-agnostically (see metrics.py).
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans

from config import KMEANS_K, KMEANS_MIN_REGION_AREA, SEED


def watershed_segmentation(image: np.ndarray) -> list[np.ndarray]:
    """
    Marker-based watershed segmentation.

    Pipeline: grayscale -> Gaussian blur -> Otsu threshold -> morphological
    opening -> distance transform for sure foreground -> dilation for sure
    background -> connected-component markers -> watershed.

    Why this underperforms on COCO, stated plainly because an interviewer will
    ask: Otsu assumes a bimodal intensity histogram, i.e. foreground objects
    separable from background by brightness alone. A natural photograph of a
    crowded street has no such split. The distance transform then merges
    touching objects into one basin, so the method systematically UNDER-counts
    exactly where the project cares most — dense scenes.

    Args:
        image: RGB uint8 array, shape (H, W, 3).

    Returns:
        List of boolean masks, one per detected region. Background and the
        watershed boundary label are excluded.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1          # background becomes 1, not 0
    markers[unknown == 255] = 0    # unknown region

    markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), markers)

    masks = []
    for label in np.unique(markers):
        if label in (-1, 1):       # boundary, background
            continue
        m = markers == label
        if m.sum() >= KMEANS_MIN_REGION_AREA:
            masks.append(m)
    return masks


def kmeans_color_segmentation(image: np.ndarray, k: int = KMEANS_K) -> list[np.ndarray]:
    """
    KMeans colour clustering followed by per-cluster connected components.

    Why this over-counts, again because it will be asked: colour clustering has
    no object prior at all. A single striped shirt splits across several clusters
    and each fragment becomes its own "instance", while a uniform wall fragments
    into dozens of regions on lighting gradients alone. The method reliably
    returns an order of magnitude more regions than there are objects.

    KMeans is fitted on a pixel subsample (not every pixel) for tractability;
    the subsample is drawn under the global SEED so the result is reproducible.

    Args:
        image: RGB uint8 array, shape (H, W, 3).
        k: Number of colour clusters.

    Returns:
        List of boolean masks, one per connected region above the area floor.
    """
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3).astype(np.float32)

    # Fit on at most 20k pixels, then predict the rest. Fitting on every pixel of
    # a 640x480 image is ~300k samples and dominates the runtime for no gain.
    rng = np.random.default_rng(SEED)
    n_fit = min(20_000, pixels.shape[0])
    fit_idx = rng.choice(pixels.shape[0], size=n_fit, replace=False)

    km = KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=SEED)
    km.fit(pixels[fit_idx])
    label_map = km.predict(pixels).reshape(h, w)

    masks = []
    for cluster_id in range(k):
        cluster_mask = np.uint8(label_map == cluster_id) * 255
        n_labels, comp, stats, _ = cv2.connectedComponentsWithStats(cluster_mask)
        for i in range(1, n_labels):   # 0 is background of this cluster mask
            if stats[i, cv2.CC_STAT_AREA] >= KMEANS_MIN_REGION_AREA:
                masks.append(comp == i)
    return masks


# ── Backwards-compatible count-only wrappers ──────────────────────────────────
# The notebooks call these. They now derive the count from the masks, so the
# count and the IoU can never again disagree about what was detected.

def watershed_count(image: np.ndarray) -> int:
    return len(watershed_segmentation(image))


def kmeans_count(image: np.ndarray, k: int = KMEANS_K) -> int:
    return len(kmeans_color_segmentation(image, k=k))
