"""
evaluate.py — the unified evaluation harness.

This is the file whose absence caused every integrity problem in the first
version of this project. Previously each phase evaluated itself, in its own
notebook, on its own sample, with its own metric definitions — and then the
README printed the rows side by side as though they were one experiment. They
were not: Phase 2 reported IoU over 50 images from a 10-epoch model on CPU,
Phase 3 reported accuracy over 100 images from a 4-epoch model on a T4.

Here, all four methods are scored:
  * on the SAME held-out test images,
  * with the SAME metric definitions (metrics.py),
  * on the SAME device, in one process, in one run,
  * writing ONE results file that every table and figure is generated from.

Run from the project root:
    .venv/bin/python src/evaluate.py
"""

from __future__ import annotations

import json
import platform
import random
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline import kmeans_color_segmentation, watershed_segmentation
from config import (
    ANN_FILE,
    BEST_WEIGHTS,
    CONF,
    COUNT_TOLERANCE,
    EXPERIMENT_VERSION,
    IMG_DIR,
    IOU_MATCH_MODE,
    IOU_NMS,
    METRICS_DIR,
    SEED,
)
from hybrid import hybrid_predict, yolo_masks
from metrics import aggregate, count_metrics, match_instances


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_gt_masks(coco: COCO, img_id: int) -> list[np.ndarray]:
    """Ground-truth instance masks for one image, crowd regions excluded."""
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
    return [coco.annToMask(a).astype(bool) for a in coco.loadAnns(ann_ids)]


def score_one(pred_masks, pred_count, gt_masks, elapsed_ms) -> dict:
    """Apply the shared metric definitions to one prediction."""
    _, mean_iou = match_instances(pred_masks, gt_masks)
    rec = count_metrics(pred_count, len(gt_masks))
    rec["mean_iou"] = round(mean_iou, 4)
    rec["n_masks"] = len(pred_masks)
    rec["inference_ms"] = round(elapsed_ms, 2)
    return rec


def main() -> None:
    seed_everything()
    device = pick_device()

    split_path = METRICS_DIR / "data_split.json"
    if not split_path.exists():
        sys.exit(f"Missing {split_path}. Run src/prepare_yolo_data.py first.")
    test_ids = json.loads(split_path.read_text())["test"]

    if not BEST_WEIGHTS.exists():
        sys.exit(f"Missing {BEST_WEIGHTS}. Run src/train.py first.")

    print("=" * 70)
    print(f"  Unified evaluation  ·  v{EXPERIMENT_VERSION}  ·  device={device}")
    print(f"  test images={len(test_ids)}  seed={SEED}  tolerance=+/-{COUNT_TOLERANCE}")
    print("=" * 70)

    coco = COCO(str(ANN_FILE))

    from ultralytics import YOLO
    model = YOLO(str(BEST_WEIGHTS))
    model.to(device)

    # Warm-up. The first inference on MPS pays kernel compilation — in the old
    # results that single 12.7-second cold start landed inside the reported mean
    # and dragged it from ~96 ms to 1650 ms.
    first = coco.loadImgs(test_ids[0])[0]
    import cv2
    warm = cv2.cvtColor(cv2.imread(str(IMG_DIR / first["file_name"])), cv2.COLOR_BGR2RGB)
    for _ in range(3):
        yolo_masks(model, warm, CONF, IOU_NMS)

    records: dict[str, list[dict]] = {
        "watershed": [], "kmeans": [], "yolo": [], "hybrid": []
    }
    hybrid_gate = {"dense": 0, "blended": 0}

    for i, img_id in enumerate(test_ids, 1):
        info = coco.loadImgs(img_id)[0]
        image = cv2.cvtColor(
            cv2.imread(str(IMG_DIR / info["file_name"])), cv2.COLOR_BGR2RGB
        )
        gt = load_gt_masks(coco, img_id)

        t = time.perf_counter()
        m = watershed_segmentation(image)
        records["watershed"].append(
            score_one(m, len(m), gt, (time.perf_counter() - t) * 1000))

        t = time.perf_counter()
        m = kmeans_color_segmentation(image)
        records["kmeans"].append(
            score_one(m, len(m), gt, (time.perf_counter() - t) * 1000))

        t = time.perf_counter()
        m = yolo_masks(model, image, CONF, IOU_NMS)
        records["yolo"].append(
            score_one(m, len(m), gt, (time.perf_counter() - t) * 1000))

        t = time.perf_counter()
        h = hybrid_predict(model, image)
        records["hybrid"].append(
            score_one(h["masks"], h["count"], gt, (time.perf_counter() - t) * 1000))
        records["hybrid"][-1]["dense"] = h["dense"]
        records["hybrid"][-1]["blended"] = h["blended"]
        records["hybrid"][-1]["edge_density"] = h["edge_density"]
        hybrid_gate["dense"] += int(h["dense"])
        hybrid_gate["blended"] += int(h["blended"])

        records["watershed"][-1]["img_id"] = img_id
        records["kmeans"][-1]["img_id"] = img_id
        records["yolo"][-1]["img_id"] = img_id
        records["hybrid"][-1]["img_id"] = img_id

        if i % 10 == 0 or i == len(test_ids):
            print(f"  [{i}/{len(test_ids)}] evaluated")

    summary = {m: aggregate(r) for m, r in records.items()}
    summary["hybrid"]["dense_gate_fired"] = hybrid_gate["dense"]
    summary["hybrid"]["watershed_blend_applied"] = hybrid_gate["blended"]

    import ultralytics
    out = {
        "provenance": {
            "experiment_version": EXPERIMENT_VERSION,
            "date": date.today().isoformat(),
            "seed": SEED,
            "device": device,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "weights": str(BEST_WEIGHTS.name),
            "n_test_images": len(test_ids),
            "iou_match_mode": IOU_MATCH_MODE,
            "count_tolerance": COUNT_TOLERANCE,
            "conf": CONF,
            "iou_nms": IOU_NMS,
            "training": json.loads((METRICS_DIR / "training_provenance.json").read_text())
                if (METRICS_DIR / "training_provenance.json").exists() else None,
            "note": (
                "All four methods scored in this single run on the same test "
                "images with the same metric definitions."
            ),
        },
        "summary": summary,
        "per_image": records,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    res_path = METRICS_DIR / "unified_results.json"
    res_path.write_text(json.dumps(out, indent=2))

    # The results table, generated — never hand-written.
    csv_path = METRICS_DIR / "results_table.csv"
    rows = ["Method,Type,Mean IoU,Count Acc% (+/-3),MAE,Median Inference (ms)"]
    labels = {
        "watershed": ("Watershed", "Classical"),
        "kmeans": (f"KMeans", "Classical"),
        "yolo": ("YOLOv8s-seg (fine-tuned)", "Deep Learning"),
        "hybrid": ("Hybrid (density-aware)", "Hybrid DL+Classical"),
    }
    for key in ["watershed", "kmeans", "yolo", "hybrid"]:
        s = summary[key]
        name, typ = labels[key]
        rows.append(
            f"{name},{typ},{s['mean_iou']},{s['count_accuracy_pct']},"
            f"{s['mae']},{s['median_inference_ms']}"
        )
    csv_path.write_text("\n".join(rows) + "\n")

    print("\n" + "=" * 70)
    print(f"{'Method':<28}{'IoU':>8}{'Acc%':>8}{'MAE':>8}{'ms':>10}")
    print("-" * 70)
    for key in ["watershed", "kmeans", "yolo", "hybrid"]:
        s = summary[key]
        print(f"{labels[key][0]:<28}{s['mean_iou']:>8.4f}"
              f"{s['count_accuracy_pct']:>8.1f}{s['mae']:>8.2f}"
              f"{s['median_inference_ms']:>10.1f}")
    print("=" * 70)
    print(f"  hybrid: dense gate fired on {hybrid_gate['dense']}/{len(test_ids)}, "
          f"watershed blend applied {hybrid_gate['blended']}")
    print(f"\n  -> {res_path}")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
