---
title: High-Density Object Segmentation
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
---

Instance segmentation and object counting in crowded scenes. A fine-tuned
YOLOv8s-seg model runs **entirely in the browser** via ONNX Runtime Web — no
server, no upload.

Compares the detector alone against a density-aware second pass gated on Canny
edge density. Measured on 50 held-out COCO val2017 images: count accuracy rises
from 52% to 64%, MAE falls from 4.28 to 3.30.

Source, ablation and 18-page report:
https://github.com/Zakuroooo/high-density-object-segmentation
