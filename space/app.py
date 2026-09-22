"""
Gradio demo for High-Density Object Segmentation.

Deliberately self-contained: a Space should not need the research repo's package
layout to boot. The thresholds below are copied from `src/config.py` and the
gating logic mirrors `src/hybrid.py`; the repo remains the source of truth and
is linked from the UI.

Source: https://github.com/Zakuroooo/high-density-object-segmentation
"""

from __future__ import annotations

import time

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

# ── Thresholds (mirror src/config.py) ─────────────────────────────────────────
CONF, IOU_NMS = 0.25, 0.45
DENSE_CONF, DENSE_IOU_NMS = 0.15, 0.35
EDGE_DENSITY_THRESHOLD, DENSE_COUNT_THRESHOLD = 0.08, 10
WATERSHED_TRUST_RATIO, YOLO_WEIGHT, WATERSHED_WEIGHT = 1.20, 0.8, 0.2
MIN_REGION_AREA = 100

MODEL = YOLO("yolov8s-seg-dense.pt")

PALETTE = np.array([
    [42, 120, 214], [235, 104, 52], [27, 175, 122], [237, 161, 0],
    [232, 123, 164], [0, 131, 0], [74, 58, 167], [227, 73, 72],
], dtype=np.uint8)


def overlay(image: np.ndarray, masks: list[np.ndarray], alpha: float = 0.55) -> np.ndarray:
    """Paint instance masks over the image, one palette hue per instance."""
    out = image.copy()
    for i, m in enumerate(masks):
        colour = PALETTE[i % len(PALETTE)]
        out[m] = (alpha * colour + (1 - alpha) * out[m]).astype(np.uint8)
        # A thin outline keeps touching instances distinguishable, which is the
        # entire difficulty in a crowded scene.
        contours, _ = cv2.findContours(
            m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, colour.tolist(), 2)
    return out


def edge_density(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges)) / edges.size


def watershed_masks(image: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), markers)
    return [markers == l for l in np.unique(markers)
            if l not in (-1, 1) and (markers == l).sum() >= MIN_REGION_AREA]


def yolo_masks(image: np.ndarray, conf: float, iou: float) -> list[np.ndarray]:
    res = MODEL.predict(image, conf=conf, iou=iou, verbose=False)[0]
    if res.masks is None:
        return []
    h, w = image.shape[:2]
    return [cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
            for m in res.masks.data.cpu().numpy()]


def run(image: np.ndarray):
    if image is None:
        return None, None, None, "Upload an image to begin."

    t0 = time.perf_counter()
    y = yolo_masks(image, CONF, IOU_NMS)
    t_yolo = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    w = watershed_masks(image)
    t_ws = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    ed = edge_density(image)
    dense = (ed > EDGE_DENSITY_THRESHOLD) or (len(y) >= DENSE_COUNT_THRESHOLD)
    h_masks = y
    if dense:
        relaxed = yolo_masks(image, DENSE_CONF, DENSE_IOU_NMS)
        if len(relaxed) > len(h_masks):
            h_masks = relaxed
    count = len(h_masks)
    blended = False
    if dense and count > 0 and len(w) > WATERSHED_TRUST_RATIO * count:
        count = int(round(YOLO_WEIGHT * count + WATERSHED_WEIGHT * len(w)))
        blended = True
    t_hy = (time.perf_counter() - t0) * 1000 + t_yolo

    report = f"""
### Result

| Method | Objects found | Time |
|---|---|---|
| Watershed (classical) | **{len(w)}** | {t_ws:.0f} ms |
| YOLOv8s-seg (fine-tuned) | **{len(y)}** | {t_yolo:.0f} ms |
| Hybrid (density-aware) | **{count}** | {t_hy:.0f} ms |

**Canny edge density:** `{ed:.4f}` — threshold is `{EDGE_DENSITY_THRESHOLD}`
**Density gate:** {"**fired** — the relaxed detector pass ran" if dense else "did not fire — ordinary scene, plain YOLO pass"}
**Watershed cross-check:** {"**applied** — watershed found substantially more regions, so the count estimate was raised" if blended else "not applied"}

{"> The hybrid reports **" + str(count) + "** objects but draws **" + str(len(h_masks)) + "** masks. When watershed finds more regions than the detector, the method raises its *count* without being able to say where those extra objects are — so both numbers are shown rather than one implying a localisation that was never produced." if count != len(h_masks) else ""}
"""
    return (overlay(image, w), overlay(image, y), overlay(image, h_masks), report)


DESCRIPTION = """
# High-Density Object Segmentation

Counting and segmenting objects in **crowded** scenes, where classical computer
vision breaks down. Upload a busy photo — a street, a shelf, a crowd — and compare
three approaches on it.

**Watershed** assumes objects separate from the background by brightness, which a
natural photograph does not do; it merges touching objects and under-counts.
**YOLOv8s-seg**, fine-tuned here on a dense subset of COCO, does far better but
still suppresses real objects when a scene gets crowded. The **hybrid** detects the
crowding first — with a cheap Canny edge-density test — and only then relaxes the
detector and cross-checks the count against watershed.

Measured on a held-out COCO test split: see the
[repository](https://github.com/Zakuroooo/high-density-object-segmentation)
for the full results table, the evaluation harness and the report.
"""

with gr.Blocks(title="High-Density Object Segmentation",
               theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="numpy", label="Input image")
            btn = gr.Button("Segment", variant="primary")
            out_md = gr.Markdown()
        with gr.Column(scale=2):
            with gr.Row():
                o1 = gr.Image(label="Watershed (classical)")
                o2 = gr.Image(label="YOLOv8s-seg (fine-tuned)")
                o3 = gr.Image(label="Hybrid (density-aware)")
    btn.click(run, inputs=inp, outputs=[o1, o2, o3, out_md])
    inp.upload(run, inputs=inp, outputs=[o1, o2, o3, out_md])

if __name__ == "__main__":
    demo.launch()
