# High-Density Object Segmentation (Phase 1 Baseline)

**Author:** Pranay Sarkar
**Institution:** Newton School of Technology  

## Project Overview
This is Phase 1 of my computer vision project. The goal here is to establish a baseline for counting and segmenting objects in highly crowded images (like retail shelves or packed crowds). 

For this first phase, I am **not** using Deep Learning. I wanted to see how classic, old-school computer vision algorithms (Watershed and KMeans) handle dense overlapping objects. To do this, I wrote a script to filter the `COCO val2017` dataset to only include images with 5 to 50 objects.

## Folder Structure
- `data/` - Holds the downloaded COCO validation images and annotations (ignored in git).
- `notebooks/` - Contains my EDA (`01_eda.ipynb`) and my baseline testing (`02_baseline_ml.ipynb`).
- `src/` - My python scripts for loading data and running the algorithms.
- `results/` - Where all the charts and JSON metrics are saved.
- `report/` - My LaTeX report containing the math and literature review.

## Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/Zakuroooo/high-density-object-segmentation.git
   cd high-density-object-segmentation
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the COCO dataset (I am only using the validation set to save space):
   ```bash
   mkdir -p data/images data/annotations
   wget http://images.cocodataset.org/zips/val2017.zip -P data/images/
   unzip data/images/val2017.zip -d data/images/
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P data/annotations/
   unzip data/annotations/annotations_trainval2017.zip -d data/annotations/
   ```

## How to Run

1. Run EDA first to see the data distribution: `jupyter notebook notebooks/01_eda.ipynb`
2. Run the Baseline Models next: `jupyter notebook notebooks/02_baseline_ml.ipynb`
3. To compile the report, `cd report` and run `pdflatex main.tex`

## Phase 1 Results Summary

I tested both algorithms on 100 dense images. The actual ground truth average was about 12 objects per image.

- **Watershed Algorithm:** Predicted an average of ~4 objects. It severely under-counted because overlapping objects don't have clear borders, so it merged them together.
- **KMeans (K=5):** Predicted an average of ~127 regions. It completely over-counted because a single object usually has multiple colors, so it shattered them into tiny fragments.

**Conclusion:** Classic algorithms struggle heavily with occlusion and scale variation. Phase 2 will require Deep Learning (like YOLO or Mask R-CNN) to actually understand the image context.
