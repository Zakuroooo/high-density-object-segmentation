"""
prepare_yolo_data.py — Prepares a COCO dense-image subset in YOLO segmentation format.

Pipeline:
    1. Load instances_val2017.json
    2. Filter dense images (5-50 objects) — same logic as Phase 1
    3. Take first 500 dense images
    4. Split 400 / 50 / 50  (train / val / test)
    5. Convert COCO polygon annotations → YOLO segmentation .txt files
    6. Copy images to data/yolo/images/{train,val,test}/
    7. Write data/yolo/coco_dense.yaml
    8. Save split summary to results/metrics/data_split.json

Run from the project root:
    python src/prepare_yolo_data.py
"""

import os
import json
import shutil
from pathlib import Path

from pycocotools.coco import COCO

# ── COCO category names in the canonical order (id 1-90, 80 used) ──────────────
COCO_CATEGORY_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Paths (relative to project root)
ANN_FILE   = os.path.join("data", "annotations", "annotations", "instances_val2017.json")
IMG_DIR    = os.path.join("data", "images", "val2017")
YOLO_ROOT  = os.path.join("data", "yolo")
YAML_PATH  = os.path.join(YOLO_ROOT, "coco_dense.yaml")
SPLIT_JSON = os.path.join("results", "metrics", "data_split.json")

TOTAL_SUBSET = 500
TRAIN_SIZE   = 400
VAL_SIZE     = 50
TEST_SIZE    = 50


def get_dense_images(coco: COCO, min_obj: int = 5, max_obj: int = 50) -> list:
    """
    Filter COCO images to those with annotation count in [min_obj, max_obj].

    Args:
        coco (COCO): Loaded pycocotools COCO object.
        min_obj (int): Minimum objects per image (inclusive).
        max_obj (int): Maximum objects per image (inclusive).

    Returns:
        list[int]: Sorted list of qualifying image IDs.
    """
    img_ids = coco.getImgIds()
    dense_ids = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        if min_obj <= len(ann_ids) <= max_obj:
            dense_ids.append(img_id)
    return sorted(dense_ids)


def polygon_to_yolo(segmentation: list, img_w: int, img_h: int) -> list:
    """
    Convert a COCO polygon segmentation to normalised YOLO coordinates.

    For multi-part polygons (rare) only the largest part is used.

    Args:
        segmentation (list): List of polygon point lists from COCO annotation.
        img_w (int): Image width in pixels.
        img_h (int): Image height in pixels.

    Returns:
        list[float]: Flat list of normalised (x, y) pairs  [x1, y1, x2, y2, ...].
                     Returns empty list if segmentation is empty.
    """
    if not segmentation:
        return []

    # Pick the largest polygon (most points) for multi-part annotations
    polygon = max(segmentation, key=len)

    points = []
    for i in range(0, len(polygon) - 1, 2):
        x = max(0.0, min(1.0, polygon[i]     / img_w))
        y = max(0.0, min(1.0, polygon[i + 1] / img_h))
        points.extend([x, y])

    return points


def build_category_map(coco: COCO) -> dict:
    """
    Build a mapping from COCO category ID → YOLO class index (0-based).

    The YOLO class order follows COCO_CATEGORY_NAMES list defined above.

    Args:
        coco (COCO): Loaded pycocotools COCO object.

    Returns:
        dict: {coco_cat_id: yolo_class_idx}
    """
    cats = coco.loadCats(coco.getCatIds())
    name_to_yolo = {name: idx for idx, name in enumerate(COCO_CATEGORY_NAMES)}
    cat_map = {}
    for cat in cats:
        yolo_idx = name_to_yolo.get(cat["name"])
        if yolo_idx is not None:
            cat_map[cat["id"]] = yolo_idx
    return cat_map


def write_yolo_label(label_path: str, anns: list, img_w: int, img_h: int,
                     cat_map: dict) -> None:
    """
    Write a YOLO segmentation label file for a single image.

    Each line: <class_id> <x1> <y1> <x2> <y2> ... (space-separated, normalised)

    Args:
        label_path (str): Destination .txt file path.
        anns (list[dict]): COCO annotation dicts for this image.
        img_w (int): Image width in pixels.
        img_h (int): Image height in pixels.
        cat_map (dict): {coco_cat_id: yolo_class_idx}
    """
    lines = []
    for ann in anns:
        if ann.get("iscrowd", 0):
            continue
        seg = ann.get("segmentation", [])
        if not seg or isinstance(seg, dict):           # skip RLE masks
            continue
        yolo_class = cat_map.get(ann["category_id"])
        if yolo_class is None:
            continue
        coords = polygon_to_yolo(seg, img_w, img_h)
        if len(coords) < 6:                            # need at least 3 points
            continue
        coord_str = " ".join(f"{v:.6f}" for v in coords)
        lines.append(f"{yolo_class} {coord_str}")

    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w") as f:
        f.write("\n".join(lines))


def process_split(split_name: str, img_ids: list, coco: COCO,
                  cat_map: dict) -> None:
    """
    Copy images and write YOLO label files for a single data split.

    Images → data/yolo/images/<split_name>/
    Labels → data/yolo/labels/<split_name>/

    Args:
        split_name (str): One of 'train', 'val', 'test'.
        img_ids (list[int]): Image IDs for this split.
        coco (COCO): Loaded pycocotools COCO object.
        cat_map (dict): {coco_cat_id: yolo_class_idx}
    """
    img_out_dir   = os.path.join(YOLO_ROOT, "images", split_name)
    label_out_dir = os.path.join(YOLO_ROOT, "labels", split_name)
    os.makedirs(img_out_dir,   exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    print(f"\n  Processing {split_name} split ({len(img_ids)} images)...")
    for i, img_id in enumerate(img_ids, 1):
        img_info = coco.loadImgs(img_id)[0]
        fname    = img_info["file_name"]
        src_path = os.path.join(IMG_DIR, fname)
        dst_path = os.path.join(img_out_dir, fname)

        # Copy image
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"    [WARNING] Image not found: {src_path}")

        # Write label file
        ann_ids    = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns       = coco.loadAnns(ann_ids)
        label_name = os.path.splitext(fname)[0] + ".txt"
        label_path = os.path.join(label_out_dir, label_name)
        write_yolo_label(label_path, anns,
                         img_info["width"], img_info["height"], cat_map)

        if i % 50 == 0 or i == len(img_ids):
            print(f"    [{split_name}] {i}/{len(img_ids)} images done.")


def write_yaml(yaml_path: str) -> None:
    """
    Write the YOLO dataset YAML configuration file.

    Args:
        yaml_path (str): Path to write coco_dense.yaml.
    """
    names_str = "\n".join(f"  - {name}" for name in COCO_CATEGORY_NAMES)
    content = f"""\
# coco_dense.yaml  — YOLO dataset config for dense COCO subset (Phase 2)
path: ../data/yolo
train: images/train
val: images/val
test: images/test

nc: {len(COCO_CATEGORY_NAMES)}
names:
{names_str}
"""
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"\n  YAML written → {yaml_path}")


def main():
    """
    Main entry point for YOLO data preparation.

    Loads COCO, filters dense images, splits into train/val/test,
    converts annotations to YOLO format, and writes the YAML config.
    """
    print("=" * 60)
    print("  COCO → YOLO Data Preparation (Phase 2)")
    print("=" * 60)

    # 1. Load COCO annotations
    print(f"\n[1] Loading COCO annotations from {ANN_FILE} ...")
    try:
        coco = COCO(ANN_FILE)
    except FileNotFoundError:
        print(f"  [ERROR] Annotation file not found: {ANN_FILE}")
        print("  Please download COCO val2017 annotations first.")
        raise

    # 2. Filter dense images
    print("\n[2] Filtering dense images (5-50 objects per image)...")
    dense_ids = get_dense_images(coco)
    print(f"  Found {len(dense_ids)} dense images.")

    # 3. Take first 500
    subset_ids = dense_ids[:TOTAL_SUBSET]
    print(f"\n[3] Using first {TOTAL_SUBSET} dense images.")

    # 4. Split 400/50/50
    train_ids = subset_ids[:TRAIN_SIZE]
    val_ids   = subset_ids[TRAIN_SIZE: TRAIN_SIZE + VAL_SIZE]
    test_ids  = subset_ids[TRAIN_SIZE + VAL_SIZE:]
    print(f"\n[4] Split: train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    # Save split IDs
    os.makedirs(os.path.dirname(SPLIT_JSON), exist_ok=True)
    split_summary = {
        "train": train_ids,
        "val":   val_ids,
        "test":  test_ids,
        "total_dense_images": len(dense_ids),
        "subset_size": TOTAL_SUBSET,
    }
    with open(SPLIT_JSON, "w") as f:
        json.dump(split_summary, f, indent=2)
    print(f"  Split summary saved → {SPLIT_JSON}")

    # 5. Build category map
    cat_map = build_category_map(coco)
    print(f"\n[5] Built category map for {len(cat_map)} COCO categories.")

    # 6 + convert all splits
    print("\n[6] Converting annotations to YOLO format & copying images...")
    process_split("train", train_ids, coco, cat_map)
    process_split("val",   val_ids,   coco, cat_map)
    process_split("test",  test_ids,  coco, cat_map)

    # 7. Write YAML
    print("\n[7] Writing YOLO dataset YAML...")
    write_yaml(YAML_PATH)

    print("\n" + "=" * 60)
    print("  Data preparation complete!")
    print(f"  Train : {len(train_ids)} images")
    print(f"  Val   : {len(val_ids)} images")
    print(f"  Test  : {len(test_ids)} images")
    print(f"  YAML  : {YAML_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
