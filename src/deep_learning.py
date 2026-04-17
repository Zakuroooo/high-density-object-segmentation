"""
deep_learning.py — YOLOv8 training, inference, and evaluation utilities
for Phase 2 of the High-Density Object Segmentation project.

Functions:
    get_device()            — Detect MPS / CUDA / CPU
    train_yolo()            — Fine-tune YOLOv8s instance segmentation
    run_inference()         — Batch inference with progress reporting
    compute_metrics()       — Per-image IoU, count accuracy, timing
    plot_training_curves()  — Visualise loss and mAP from results.csv
"""

import os
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# 4.1  Device detection
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> str:
    """
    Automatically detect the best available compute device.

    Priority: Apple MPS (M1/M2) → NVIDIA CUDA → CPU.
    Prints a message describing which device will be used.

    Returns:
        str: Device string compatible with Ultralytics — one of
             'mps', 'cuda', 'cpu'.
    """
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("[Device] Apple MPS (Metal Performance Shaders) detected — using M2 GPU.")
            return "mps"  # using mps because my mac has m2 chip
        elif torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[Device] CUDA GPU detected: {gpu_name}")
            return "cuda"
        else:
            print("[Device] No GPU detected — falling back to CPU.")
            return "cpu"
    except ImportError:
        print("[Device] PyTorch not installed — falling back to CPU.")
        return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# 4.2  Training
# ─────────────────────────────────────────────────────────────────────────────

def train_yolo(
    data_yaml: str,
    epochs: int = 10,
    imgsz: int = 640,
    model_name: str = "yolov8s-seg.pt",
):
    """
    Fine-tune YOLOv8s-seg on the prepared COCO dense subset.

    Loads the pretrained YOLOv8s-seg weights and fine-tunes them for
    instance segmentation on our 400-image training set.  Training
    artefacts (weights, results.csv, confusion matrix, etc.) are saved
    to runs/segment/phase2_training/.

    Hyperparameters:
        - batch=8       : fits comfortably in 16 GB unified memory (M2)
        - workers=4     : safe for macOS dataloader
        - patience=5    : early stopping after 5 epochs of no improvement
        - dropout=0.1   : light regularisation to avoid overfitting on 400 imgs
        - weight_decay=0.0005 : L2 regularisation (standard YOLO default)

    Args:
        data_yaml (str): Path to coco_dense.yaml dataset config file.
        epochs (int): Maximum number of training epochs. Default 10.
        imgsz (int): Input image size (square). Default 640.
        model_name (str): Pretrained model checkpoint name or path.
                          Default 'yolov8s-seg.pt'.

    Returns:
        ultralytics.YOLO: The trained model object with loaded best weights.
    """
    from ultralytics import YOLO

    device = get_device()
    print(f"\n[Training] Starting YOLOv8 fine-tuning on device: {device}")
    print(f"  Data YAML : {data_yaml}")
    print(f"  Epochs    : {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Model     : {model_name}\n")

    model = YOLO(model_name)

    model.train(
        data=data_yaml,
        epochs=epochs, # 10 epochs was enough, more caused overfitting
        imgsz=imgsz,
        batch=2,
        workers=0,
        patience=5,
        dropout=0.1,
        weight_decay=0.0005,
        device=device,
        project="runs/segment",
        name="phase2_training",
        exist_ok=True,
        verbose=True,
    )

    # Reload best weights
    best_weights = os.path.join("runs", "segment", "phase2_training", "weights", "best.pt")
    if os.path.exists(best_weights):
        print(f"\n[Training] Loading best weights from {best_weights}")
        model = YOLO(best_weights)
    else:
        print("[Training] best.pt not found — using last checkpoint.")

    print("[Training] Training complete.\n")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4.3  Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, image_paths: list, conf: float = 0.25) -> list:
    """
    Run YOLOv8 instance segmentation inference on a list of images.

    Saves annotated (predictions overlaid) images to
    results/figures/yolo_predictions/.

    Args:
        model (ultralytics.YOLO): Trained or pretrained YOLO model.
        image_paths (list[str]): List of absolute or relative image file paths.
        conf (float): Confidence threshold for predictions. Default 0.25.

    Returns:
        list: List of Ultralytics Results objects, one per image.
    """
    out_dir = os.path.join("results", "figures", "yolo_predictions")
    os.makedirs(out_dir, exist_ok=True)

    device = get_device()
    all_results = []

    print(f"[Inference] Running on {len(image_paths)} images (conf={conf}) ...")
    for i, img_path in enumerate(image_paths, 1):
        try:
            results = model.predict(
                source=img_path,
                conf=conf, # conf=0.25 worked better than 0.5 after some testing
                device=device,
                verbose=False,
            )
            all_results.extend(results)

            # Save annotated image
            for r in results:
                annotated = r.plot()          # numpy BGR array
                fname     = Path(img_path).stem + "_pred.jpg"
                save_path = os.path.join(out_dir, fname)
                annotated_rgb = annotated[:, :, ::-1]   # BGR → RGB
                Image.fromarray(annotated_rgb).save(save_path)

        except Exception as e:
            print(f"  [WARNING] Inference failed for {img_path}: {e}")
            all_results.append(None)

        if i % 10 == 0 or i == len(image_paths):
            print(f"  [Inference] {i}/{len(image_paths)} images processed.")

    print(f"[Inference] Done. Annotated images saved to {out_dir}\n")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# 4.4  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _box_iou(box1: list, box2: list) -> float:
    """
    Compute IoU between two axis-aligned bounding boxes.

    Args:
        box1 (list): [x1, y1, x2, y2] (pixel coordinates).
        box2 (list): [x1, y1, x2, y2] (pixel coordinates).

    Returns:
        float: IoU score in [0, 1].
    """
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter   = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def _mean_iou_for_image(pred_boxes: list, gt_boxes: list) -> float:
    """
    Compute mean IoU between predicted and ground-truth boxes for one image.

    Each predicted box is matched to the ground-truth box with the highest
    IoU.  Unmatched predictions and GT boxes contribute 0.

    Args:
        pred_boxes (list): Predicted boxes [[x1,y1,x2,y2], ...] in pixels.
        gt_boxes (list):   GT boxes [[x1,y1,x2,y2], ...] in pixels.

    Returns:
        float: Mean IoU for this image.
    """
    if not pred_boxes or not gt_boxes:
        return 0.0

    ious = []
    for pb in pred_boxes:
        best = max(_box_iou(pb, gb) for gb in gt_boxes)
        ious.append(best)

    # Also penalise for GT boxes with no matching prediction
    n_missed = max(0, len(gt_boxes) - len(pred_boxes))
    ious.extend([0.0] * n_missed)

    return float(np.mean(ious))


def compute_metrics(results_list: list, coco, img_ids: list) -> pd.DataFrame:
    """
    Compute per-image evaluation metrics for YOLOv8 predictions.

    For each image records:
        - img_id            : COCO image ID
        - gt_count          : Ground-truth object count
        - pred_count        : Predicted object count
        - count_error       : abs(pred - gt)
        - within_3          : 1 if count_error <= 3, else 0
        - mean_iou          : Mean IoU between predicted and GT boxes
        - inference_ms      : Inference time in milliseconds (from YOLO speed)

    Results are saved to results/metrics/dl_results.json.

    Args:
        results_list (list): List of Ultralytics Results objects from run_inference().
        coco: pycocotools COCO object (for GT bounding boxes).
        img_ids (list[int]): COCO image IDs corresponding to results_list entries.

    Returns:
        pd.DataFrame: DataFrame with one row per image and the metrics above.
    """
    records = []

    for i, (result, img_id) in enumerate(zip(results_list, img_ids)):
        # ── Ground-truth ────────────────────────────────────────────────────
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = coco.loadAnns(ann_ids)
        gt_count = len(anns)
        gt_boxes = [[a["bbox"][0],
                     a["bbox"][1],
                     a["bbox"][0] + a["bbox"][2],
                     a["bbox"][1] + a["bbox"][3]]
                    for a in anns if a["bbox"][2] > 0 and a["bbox"][3] > 0]

        # ── Predictions ─────────────────────────────────────────────────────
        if result is None:
            pred_count = 0
            pred_boxes = []
            inf_ms     = 0.0
        else:
            boxes = result.boxes
            pred_count = len(boxes) if boxes is not None else 0
            pred_boxes = (boxes.xyxy.cpu().numpy().tolist()
                          if boxes is not None and len(boxes) > 0 else [])
            # speed dict: {'preprocess': ms, 'inference': ms, 'postprocess': ms}
            speed  = result.speed if hasattr(result, "speed") else {}
            inf_ms = speed.get("inference", 0.0)

        # ── Metrics ─────────────────────────────────────────────────────────
        mean_iou    = _mean_iou_for_image(pred_boxes, gt_boxes)
        count_error = abs(pred_count - gt_count)

        records.append({
            "img_id"       : img_id,
            "gt_count"     : gt_count,
            "pred_count"   : pred_count,
            "count_error"  : count_error,
            "within_3"     : int(count_error <= 3),
            "mean_iou"     : round(mean_iou, 4),
            "inference_ms" : round(inf_ms, 2),
        })

    df = pd.DataFrame(records)

    # Save
    out_path = os.path.join("results", "metrics", "dl_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_json(out_path, orient="records", indent=2)
    print(f"[Metrics] Results saved → {out_path}")

    # Summary
    print("\n── Evaluation Summary ─────────────────────────────────────────")
    print(f"  Mean GT count      : {df['gt_count'].mean():.2f}")
    print(f"  Mean pred count    : {df['pred_count'].mean():.2f}")
    print(f"  Mean count error   : {df['count_error'].mean():.2f}")
    print(f"  Count accuracy±3   : {df['within_3'].mean() * 100:.1f}%")
    print(f"  Mean IoU           : {df['mean_iou'].mean():.4f}")
    print(f"  Mean inference (ms): {df['inference_ms'].mean():.1f}")
    print("──────────────────────────────────────────────────────────────\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.5  Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(results_csv_path: str) -> None:
    """
    Plot YOLOv8 training and validation loss / mAP curves from results.csv.

    Generates two side-by-side charts:
        Left  — train/val box + seg loss vs epoch
        Right — mAP50 and mAP50-95 vs epoch

    The figure is saved to results/figures/training_curves.png.

    Args:
        results_csv_path (str): Path to the YOLO results.csv file produced
                                during training (e.g. runs/segment/
                                phase2_training/results.csv).

    Raises:
        FileNotFoundError: If results_csv_path does not exist.
    """
    if not os.path.exists(results_csv_path):
        raise FileNotFoundError(f"results.csv not found at: {results_csv_path}")

    df = pd.read_csv(results_csv_path)
    df.columns = [c.strip() for c in df.columns]   # strip whitespace from column names

    epochs = df.get("epoch", pd.Series(range(len(df)))) + 1

    # ─── Figure setup ────────────────────────────────────────────────────────
    sns.set_theme(style="darkgrid", palette="muted")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("YOLOv8s-seg Training Curves — Phase 2", fontsize=14, fontweight="bold")

    # ─── Left: Loss curves ───────────────────────────────────────────────────
    ax = axes[0]
    loss_cols = {
        "train/box_loss"    : ("Train Box Loss",    "#E74C3C"),
        "train/seg_loss"    : ("Train Seg Loss",    "#F39C12"),
        "val/box_loss"      : ("Val Box Loss",      "#2ECC71"),
        "val/seg_loss"      : ("Val Seg Loss",      "#3498DB"),
    }
    for col, (label, color) in loss_cols.items():
        if col in df.columns:
            ax.plot(epochs, df[col], label=label, color=color, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Val Loss")
    ax.legend(fontsize=9)

    # ─── Right: mAP curves ───────────────────────────────────────────────────
    ax = axes[1]
    map_cols = {
        "metrics/mAP50(B)"    : ("mAP50 (Boxes)",   "#9B59B6"),
        "metrics/mAP50-95(B)" : ("mAP50-95 (Boxes)","#1ABC9C"),
        "metrics/mAP50(M)"    : ("mAP50 (Masks)",   "#E67E22"),
        "metrics/mAP50-95(M)" : ("mAP50-95 (Masks)","#2980B9"),
    }
    for col, (label, color) in map_cols.items():
        if col in df.columns:
            ax.plot(epochs, df[col], label=label, color=color, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("mAP50 and mAP50-95 vs Epoch")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join("results", "figures", "training_curves.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Curves] Training curves saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = get_device()
    print(f"deep_learning.py self-test passed — device: {device}")
