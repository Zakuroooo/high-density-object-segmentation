"""
data_loader.py — Utilities for loading and filtering COCO val2017 images
by object density for high-density segmentation research.
"""

import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image


def get_dense_images(coco, min_obj=5, max_obj=50):
    """
    Filter COCO images by annotation density.

    Args:
        coco: A pycocotools.coco.COCO object loaded with instance annotations.
        min_obj (int): Minimum number of object annotations per image (inclusive).
        max_obj (int): Maximum number of object annotations per image (inclusive).

    Returns:
        list[int]: A sorted list of image IDs whose annotation count
                   falls within [min_obj, max_obj].
    """
    img_ids = coco.getImgIds()
    dense_ids = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        count = len(ann_ids)
        if min_obj <= count <= max_obj:
            dense_ids.append(img_id)
    return sorted(dense_ids)


def load_image_and_masks(coco, img_id, img_dir):
    """
    Load an image and its COCO annotations.

    Args:
        coco: A pycocotools.coco.COCO object.
        img_id (int): The COCO image ID to load.
        img_dir (str): Relative path to the directory containing images
                       (e.g. 'data/images/val2017').

    Returns:
        tuple:
            - image (np.ndarray): The image as an RGB numpy array of shape (H, W, 3).
            - anns (list[dict]): List of COCO annotation dictionaries for this image.
    """
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, img_info['file_name'])
    image = np.array(Image.open(img_path).convert('RGB'))
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    return image, anns


def get_dataset_stats(coco):
    """
    Compute and print summary statistics for the loaded COCO dataset.

    Prints:
        - Total number of images
        - Total number of annotations (non-crowd)
        - Number of unique categories
        - Average, minimum, and maximum annotations per image

    Args:
        coco: A pycocotools.coco.COCO object.

    Returns:
        dict: A dictionary with keys 'total_images', 'total_annotations',
              'num_categories', 'avg_objects', 'min_objects', 'max_objects'.
    """
    img_ids = coco.getImgIds()
    counts = []
    total_anns = 0
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        c = len(ann_ids)
        counts.append(c)
        total_anns += c

    counts = np.array(counts)
    stats = {
        'total_images': len(img_ids),
        'total_annotations': int(total_anns),
        'num_categories': len(coco.getCatIds()),
        'avg_objects': float(np.mean(counts)),
        'min_objects': int(np.min(counts)),
        'max_objects': int(np.max(counts)),
    }

    print(f"Total images       : {stats['total_images']}")
    print(f"Total annotations  : {stats['total_annotations']}")
    print(f"Unique categories  : {stats['num_categories']}")
    print(f"Avg objects/image  : {stats['avg_objects']:.2f}")
    print(f"Min objects/image  : {stats['min_objects']}")
    print(f"Max objects/image  : {stats['max_objects']}")

    return stats


if __name__ == '__main__':
    ann_file = os.path.join('data', 'annotations', 'annotations',
                            'instances_val2017.json')
    print(f"Loading COCO annotations from {ann_file} ...")
    coco = COCO(ann_file)
    print()
    get_dataset_stats(coco)
    print()
    dense = get_dense_images(coco)
    print(f"Dense images (5-50 objects): {len(dense)}")
