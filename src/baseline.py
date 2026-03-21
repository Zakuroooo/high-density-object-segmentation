"""
baseline.py — Classical segmentation baselines for object counting
in high-density images.

Two methods implemented:
1. Watershed Segmentation (Otsu + morphological operations + watershed)
2. KMeans Colour Segmentation (colour clustering + connected components)
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


def watershed_segmentation(image):
    """
    Segment an image using the Watershed algorithm.

    Pipeline:
        1. Convert to grayscale
        2. Gaussian blur (kernel size 5)
        3. Otsu thresholding to binarise
        4. Morphological opening to remove noise
        5. Sure background via dilation
        6. Distance transform for sure foreground
        7. Watershed to label connected regions

    Args:
        image (np.ndarray): Input RGB image of shape (H, W, 3), dtype uint8.

    Returns:
        int: Number of detected object instances (excluding background).
    """
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Step 2: Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Otsu thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 4: Morphological opening (noise removal)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Step 5: Sure background (dilate the opening)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Step 6: Sure foreground via distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Step 7: Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Step 8: Label markers for watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # shift so background is 1, not 0
    markers[unknown == 255] = 0  # mark unknown as 0

    # Step 9: Apply watershed
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(img_bgr, markers)

    # Count unique labels (ignore background=1 and boundary=-1)
    unique_labels = set(markers.flatten())
    unique_labels.discard(-1)  # boundary
    unique_labels.discard(1)   # background
    num_objects = len(unique_labels)

    return num_objects


def kmeans_color_segmentation(image, k=5):
    """
    Segment an image using KMeans colour clustering.

    Pipeline:
        1. Reshape image pixels into (N, 3) feature matrix
        2. Apply KMeans clustering with k clusters
        3. Reshape cluster labels back to image shape
        4. For each cluster, run connected components analysis
        5. Sum up all connected regions across clusters (skip tiny regions)

    Args:
        image (np.ndarray): Input RGB image of shape (H, W, 3), dtype uint8.
        k (int): Number of colour clusters. Default is 5.

    Returns:
        int: Number of detected object regions across all clusters.
    """
    h, w, c = image.shape

    # Step 1: Reshape to (N, 3) and convert to float
    pixels = image.reshape(-1, 3).astype(np.float32)

    # Step 2: KMeans clustering
    kmeans = KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=42)
    labels = kmeans.fit_predict(pixels)
    label_map = labels.reshape(h, w)

    # Step 3: Connected components per cluster
    total_regions = 0
    min_region_area = 100  # ignore tiny noise regions

    for cluster_id in range(k):
        cluster_mask = np.uint8(label_map == cluster_id) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(cluster_mask)
        # Skip background label (0) and tiny regions
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_region_area:
                total_regions += 1

    return total_regions


if __name__ == '__main__':
    # Quick test with a random image
    test_img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    print(f'Watershed detected: {watershed_segmentation(test_img)} objects')
    print(f'KMeans detected:    {kmeans_color_segmentation(test_img)} objects')
    print('baseline.py test passed.')
