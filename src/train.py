"""
train.py — fine-tune YOLOv8s-seg on the dense COCO subset.

Replaces the training cell that lived only inside a notebook. The old project
trained twice, in two places, with different settings (10 epochs in Phase 2,
4 epochs in Phase 3) and then reported numbers from both models as one result.
There is now one training entry point reading one config, and the weights it
produces are the weights the evaluation loads.

Run from the project root:
    .venv/bin/python src/train.py
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BASE_MODEL,
    BATCH,
    BEST_WEIGHTS,
    EPOCHS,
    EXPERIMENT_VERSION,
    IMGSZ,
    METRICS_DIR,
    PATIENCE,
    RUNS_DIR,
    SEED,
    WEIGHTS_DIR,
    YOLO_YAML,
)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not YOLO_YAML.exists():
        sys.exit(f"Missing {YOLO_YAML}. Run src/prepare_yolo_data.py first.")

    print("=" * 70)
    print(f"  Fine-tuning {BASE_MODEL}  ·  v{EXPERIMENT_VERSION}")
    print(f"  epochs={EPOCHS}  imgsz={IMGSZ}  batch={BATCH}  device={device}  seed={SEED}")
    print("=" * 70)

    from ultralytics import YOLO

    model = YOLO(BASE_MODEL)
    model.train(
        data=str(YOLO_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        seed=SEED,
        device=device,
        project=str(RUNS_DIR),
        name="dense-seg",
        exist_ok=True,
        deterministic=True,
        val=True,
        plots=True,
        # 8 GB unified memory: workers=0 keeps the dataloader in-process rather
        # than forking copies of the dataset into memory it does not have.
        workers=0,
    )

    src = RUNS_DIR / "dense-seg" / "weights" / "best.pt"
    if not src.exists():
        sys.exit(f"Training finished but {src} is missing.")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, BEST_WEIGHTS)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "training_provenance.json").write_text(json.dumps({
        "experiment_version": EXPERIMENT_VERSION,
        "date": date.today().isoformat(),
        "base_model": BASE_MODEL,
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "seed": SEED,
        "device": device,
        "weights": str(BEST_WEIGHTS.name),
        "size_mb": round(BEST_WEIGHTS.stat().st_size / 1e6, 1),
    }, indent=2))

    print(f"\n  weights -> {BEST_WEIGHTS} "
          f"({BEST_WEIGHTS.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
