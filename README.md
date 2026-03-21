# High-Density Object Segmentation — Baseline Study

Instance segmentation in crowded, real-world images where dozens of objects overlap, vary in scale, and are partially occluded.

## Problem Statement

Segmenting individual objects in high-density scenes is one of the hardest problems in computer vision. When images contain 5–50 overlapping objects, classical methods fail because object boundaries merge under occlusion, objects span vastly different scales, and colour/intensity cues are ambiguous. This project establishes a quantitative baseline using two classical segmentation methods on a density-filtered subset of the COCO benchmark, demonstrating their limitations and motivating the use of deep learning in later phases.

## Project Structure

```
high-density-object-segmentation/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── data/                              # COCO val2017 dataset (not tracked)
│   ├── images/val2017/                # 5,000 validation images
│   └── annotations/annotations/      # Instance annotation JSONs
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory data analysis
│   └── 02_baseline_ml.ipynb           # Baseline method evaluation
├── src/
│   ├── data_loader.py                 # COCO loading & filtering utilities
│   └── baseline.py                    # Watershed & KMeans segmentation
├── results/
│   ├── figures/                       # All generated plots & visualisations
│   └── metrics/                       # JSON files with numeric results
└── report/
    ├── main.tex                       # LaTeX report
    └── refs.bib                       # BibTeX references
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/high-density-object-segmentation.git
cd high-density-object-segmentation
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Download COCO val2017 dataset

```bash
mkdir -p data/images data/annotations

# Download images (~778 MB)
wget http://images.cocodataset.org/zips/val2017.zip -P data/images/
unzip data/images/val2017.zip -d data/images/

# Download annotations (~241 MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P data/annotations/
unzip data/annotations/annotations_trainval2017.zip -d data/annotations/
```

## How to Run

### Run notebooks in order

1. **EDA notebook** (run first):
   ```bash
   cd notebooks
   jupyter notebook 01_eda.ipynb
   ```
   Generates density distribution, category charts, sample images, and saves statistics to `results/metrics/eda_stats.json`.

2. **Baseline ML notebook** (run second):
   ```bash
   jupyter notebook 02_baseline_ml.ipynb
   ```
   Evaluates Watershed and KMeans on 100 dense images, generates scatter plots, failure/success cases, and saves results to `results/metrics/baseline_results.json`.

### Compile the LaTeX report

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Key Results

| Method       | Avg Predicted Count | Avg GT Count | MAE    | Accuracy (±3) |
|-------------|--------------------:|-------------:|-------:|--------------:|
| Watershed   |                4.41 |        12.04 |   8.47 |         24.0% |
| KMeans (k=5)|              127.55 |        12.04 | 115.51 |          0.0% |

- **Watershed** under-segments: fails to split touching/occluded objects
- **KMeans** over-segments: fragments backgrounds into many small regions

## References

1. T.-Y. Lin, M. Maire, S. Belongie, et al., "Microsoft COCO: Common Objects in Context," *ECCV*, 2014.
2. P. F. Felzenszwalb and D. P. Huttenlocher, "Efficient Graph-Based Image Segmentation," *IJCV*, 59(2):167–181, 2004.
3. L. Vincent and P. Soille, "Watersheds in Digital Spaces: An Efficient Algorithm Based on Immersion Simulations," *IEEE TPAMI*, 13(6):583–598, 1991.
4. N. Otsu, "A Threshold Selection Method from Gray-Level Histograms," *IEEE Trans. SMC*, 9(1):62–66, 1979.

## License

This project is for academic/research purposes.
