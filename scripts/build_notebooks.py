"""
build_notebooks.py — generate the project's notebooks from one definition.

The notebooks are build artefacts here, not hand-edited source. That is a
deliberate reaction to how the first version went wrong: four notebooks drifted
apart, three of them were committed with their outputs stripped or never run at
all, and the Phase-3 notebook — the one the headline result came from — had zero
executed cells. Generating them from `src/` guarantees they call the same code
the evaluation does.

    .venv/bin/python scripts/build_notebooks.py       # write
    .venv/bin/jupyter nbconvert --execute --inplace notebooks/*.ipynb
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"

BOOT = """\
import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
%matplotlib inline
"""

EDA = [
    ("md", """\
# 1 — Exploratory data analysis

What "high density" means here, measured rather than asserted. Everything below
reads `data/annotations/annotations/instances_val2017.json` directly, so the
numbers in this notebook and the numbers in the report come from the same file.
"""),
    ("code", BOOT),
    ("code", """\
from pycocotools.coco import COCO
from config import ANN_FILE, MIN_OBJECTS, MAX_OBJECTS, SEED, SUBSET_SIZE
import numpy as np

coco = COCO(str(ANN_FILE))
img_ids = coco.getImgIds()
counts = np.array([len(coco.getAnnIds(imgIds=i, iscrowd=False)) for i in img_ids])

print(f"images in val2017        : {len(img_ids)}")
print(f"total non-crowd objects  : {counts.sum()}")
print(f"objects per image        : mean {counts.mean():.2f}  median {np.median(counts):.0f}  max {counts.max()}")
print(f"images with {MIN_OBJECTS}-{MAX_OBJECTS} objects : {((counts >= MIN_OBJECTS) & (counts <= MAX_OBJECTS)).sum()}")
"""),
    ("md", """\
The dense pool is the population this project samples from. Note how small the
genuinely crowded tail is — that is the reason the subset is drawn from the
5-50 band rather than from val2017 as a whole.
"""),
    ("code", """\
import matplotlib.pyplot as plt
from figures import SURFACE, INK, INK_2, GRID, SERIES

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(counts, bins=range(0, 65), color=SERIES["yolo"], edgecolor=SURFACE, linewidth=0.5)
ax.axvspan(MIN_OBJECTS, MAX_OBJECTS, color=INK_2, alpha=0.10)
ax.set_xlabel("non-crowd objects per image")
ax.set_ylabel("images")
ax.set_title(f"Object density in COCO val2017  ·  shaded = the dense band ({MIN_OBJECTS}-{MAX_OBJECTS})", color=INK)
ax.grid(axis="y", alpha=0.7); ax.set_axisbelow(True)
plt.show()
"""),
    ("code", """\
split = json.loads((ROOT / "results" / "metrics" / "data_split.json").read_text())
print("sampling :", split["sampling"])
print("seed     :", split["seed"])
print("pool     :", split["total_dense_images"], "dense images")
print("subset   :", split["subset_size"])
print("split    :", {k: len(split[k]) for k in ("train", "val", "test")})
print("train/test overlap:", set(split["train"]) & set(split["test"]) or "none")
"""),
]

RESULTS = [
    ("md", """\
# 2 — Methods and results

All four methods, run here on one image so the behaviour is visible, then the
aggregate numbers loaded from the single evaluation run.

Nothing in this notebook recomputes a published number. The table and the
figures come from `results/metrics/unified_results.json`, which
`src/evaluate.py` wrote in one pass over the held-out test split. If this
notebook and the README ever disagree, both are reading the same file and one
of them is stale.
"""),
    ("code", BOOT),
    ("code", """\
import cv2, numpy as np
from pycocotools.coco import COCO
from ultralytics import YOLO
from config import ANN_FILE, IMG_DIR, BEST_WEIGHTS, CONF, IOU_NMS
from baseline import watershed_segmentation, kmeans_color_segmentation
from hybrid import hybrid_predict, yolo_masks, edge_density

coco = COCO(str(ANN_FILE))
split = json.loads((ROOT / "results" / "metrics" / "data_split.json").read_text())
img_id = split["test"][0]
info = coco.loadImgs(img_id)[0]
image = cv2.cvtColor(cv2.imread(str(IMG_DIR / info["file_name"])), cv2.COLOR_BGR2RGB)
gt = [coco.annToMask(a).astype(bool) for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))]

model = YOLO(str(BEST_WEIGHTS))
print(f"image {img_id}  ·  {image.shape[1]}x{image.shape[0]}  ·  {len(gt)} ground-truth objects")
print(f"Canny edge density: {edge_density(image):.4f}")
"""),
    ("code", """\
ws = watershed_segmentation(image)
km = kmeans_color_segmentation(image)
yl = yolo_masks(model, image, CONF, IOU_NMS)
hy = hybrid_predict(model, image)

for name, n in [("ground truth", len(gt)), ("watershed", len(ws)),
                ("kmeans", len(km)), ("yolo", len(yl)), ("hybrid", hy["count"])]:
    print(f"{name:<14} {n}")
print(f"\\ndensity gate fired: {hy['dense']}   watershed blend applied: {hy['blended']}")
"""),
    ("md", """\
Watershed under-counts and KMeans over-counts, both badly, and both for
structural reasons rather than tuning ones — Otsu needs a bimodal intensity
histogram that a natural photograph does not have, and colour clustering has no
object prior at all. Those failures are the reason the project exists.
"""),
    ("code", """\
import matplotlib.pyplot as plt
from figures import SURFACE, INK

def overlay(img, masks, alpha=0.55):
    from figures import SERIES
    pal = np.array([[42,120,214],[235,104,52],[27,175,122],[237,161,0],
                    [232,123,164],[0,131,0],[74,58,167],[227,73,72]], dtype=np.uint8)
    out = img.copy()
    for i, m in enumerate(masks):
        out[m] = (alpha * pal[i % len(pal)] + (1 - alpha) * out[m]).astype(np.uint8)
    return out

panels = [("Ground truth", gt), ("Watershed", ws), ("YOLOv8s-seg", yl), ("Hybrid", hy["masks"])]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
for ax, (title, masks) in zip(axes, panels):
    ax.imshow(overlay(image, masks)); ax.axis("off")
    ax.set_title(f"{title}  ({len(masks)})", color=INK)
fig.tight_layout(); plt.show()
"""),
    ("md", "## Aggregate results — the single evaluation run"),
    ("code", """\
res = json.loads((ROOT / "results" / "metrics" / "unified_results.json").read_text())
p = res["provenance"]
print(f"experiment v{p['experiment_version']}  ·  {p['date']}  ·  seed {p['seed']}  ·  device {p['device']}")
print(f"test images: {p['n_test_images']}   IoU matching: {p['iou_match_mode']}   tolerance: +/-{p['count_tolerance']}")
print(f"torch {p['torch']}  ultralytics {p['ultralytics']}  weights {p['weights']}\\n")

import pandas as pd
rows = []
for k, label in [("watershed","Watershed"),("kmeans","KMeans"),
                 ("yolo","YOLOv8s-seg (fine-tuned)"),("hybrid","Hybrid (density-aware)")]:
    s = res["summary"][k]
    rows.append({"Method": label, "Mean IoU": s["mean_iou"],
                 "Count Acc %": s["count_accuracy_pct"], "MAE": s["mae"],
                 "Median ms": s["median_inference_ms"]})
pd.DataFrame(rows).set_index("Method")
"""),
    ("code", """\
from IPython.display import Image, display
for name in ["comparison.png", "count_scatter.png", "inference_time.png", "density_gate.png"]:
    display(Image(filename=str(ROOT / "results" / "figures" / name)))
"""),
]


def build(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def main() -> None:
    NB_DIR.mkdir(exist_ok=True)
    for name, cells in [("01_eda.ipynb", EDA),
                        ("02_methods_and_results.ipynb", RESULTS)]:
        path = NB_DIR / name
        nbf.write(build(cells), str(path))
        print(f"wrote {path.relative_to(ROOT)}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
