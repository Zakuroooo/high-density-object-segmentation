"""
config.py — the single source of truth for this experiment.

Every number that ends up in the README, the report or a figure is produced by a
run that read its settings from this file. Nothing is hardcoded twice.

The previous version of this project had no file like this, and that is exactly
how it ended up publishing Phase-2 IoU alongside Phase-3 accuracy as if they were
one result: two runs, two sample sizes, two models, one table.
"""

from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────────
# Set once, in seeds.seed_everything(), and used by every script.
SEED = 1337

# ── Paths (all relative to the project root) ──────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "images" / "val2017"
ANN_FILE = DATA_DIR / "annotations" / "annotations" / "instances_val2017.json"

YOLO_ROOT = DATA_DIR / "yolo"
YOLO_YAML = YOLO_ROOT / "coco_dense.yaml"

RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

RUNS_DIR = ROOT / "runs"
WEIGHTS_DIR = ROOT / "weights"
BEST_WEIGHTS = WEIGHTS_DIR / "yolov8s-seg-dense.pt"

# ── Dataset definition ────────────────────────────────────────────────────────
# "High density" is the whole premise of the project, so it gets defined once.
MIN_OBJECTS = 5
MAX_OBJECTS = 50

# The dense subset is sampled RANDOMLY under SEED. The previous version took
# `dense_ids[:500]` — the first 500 by ascending COCO image id — which is a
# non-random sample and a selection bias we cannot characterise.
SUBSET_SIZE = 500
SPLIT = {"train": 400, "val": 50, "test": 50}

# ── Training ──────────────────────────────────────────────────────────────────
BASE_MODEL = "yolov8s-seg.pt"
EPOCHS = 10
# YOLO's standard training resolution. Training runs on a Colab T4 rather than
# locally: an 8 GB M2 could only fit 512/batch-2 and drove the machine into heavy
# swap (~10 min/epoch). These are the values the published numbers come from, and
# they are recorded in the results provenance.
IMGSZ = 640
BATCH = 16
PATIENCE = 20

# ── Inference thresholds ──────────────────────────────────────────────────────
CONF = 0.25
IOU_NMS = 0.45

# Hybrid: relaxed pass used only when the scene looks dense.
DENSE_CONF = 0.15
DENSE_IOU_NMS = 0.35

# ── Hybrid gating ─────────────────────────────────────────────────────────────
# A scene is "dense" if either signal fires.
EDGE_DENSITY_THRESHOLD = 0.08
DENSE_COUNT_THRESHOLD = 10

# Watershed is only trusted to correct YOLO when it finds substantially more
# objects than YOLO did — i.e. when YOLO has probably merged or missed instances.
WATERSHED_TRUST_RATIO = 1.20
YOLO_WEIGHT = 0.8
WATERSHED_WEIGHT = 0.2

# ── Evaluation ────────────────────────────────────────────────────────────────
# Count accuracy tolerance: a prediction is "correct" if |pred - gt| <= this.
COUNT_TOLERANCE = 3

# Baselines are pixel-clustering methods with no notion of object class, so a
# class-aware IoU would be meaningless for them. Every method is therefore scored
# with the same CLASS-AGNOSTIC instance IoU under greedy one-to-one matching.
# Unmatched ground-truth instances score 0 and are included in the mean.
IOU_MATCH_MODE = "greedy_class_agnostic"

# KMeans clusters.
KMEANS_K = 5
KMEANS_MIN_REGION_AREA = 100

# ── Provenance ────────────────────────────────────────────────────────────────
# Bumped whenever a change would alter published numbers, so a stale figure is
# detectable rather than silently wrong.
EXPERIMENT_VERSION = "2.0"
