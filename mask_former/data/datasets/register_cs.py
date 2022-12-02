# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from sys import platform
CATEGORIES = [
    {
        "color": [255, 255, 255],
        "instances": False,
        "readable": "person",
        "name": "void--person",
        "evaluate": True,
    },
    {
        "color": [0, 0, 0],
        "instances": False,
        "readable": "Unlabeled",
        "name": "void--unlabeled",
        "evaluate": False,
    },
]


def _get_meta():
    stuff_classes = [k["readable"] for k in CATEGORIES if k["evaluate"]]
    print(stuff_classes)
    assert len(stuff_classes) == 1

    stuff_colors = [k["color"] for k in CATEGORIES if k["evaluate"]]
    assert len(stuff_colors) == 1

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret


def register_cs(root):
    print(root)
    root = os.path.join(root, "cs")
    meta = _get_meta()
    directory= [
        ("train", "train/images", "train/annotations"),
        ("test", "test/images", "test/annotations"),
    ]
    if platform == 'win32':
        directory = [
        ("train", "train\\images", "train\\annotations"),
        ("test", "test\\images", "test\\annotations"),
    ]
    for name, image_dirname, sem_seg_dirname in directory:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"cs_{name}"
        print("dict: ",load_sem_seg(gt_dir, image_dir, gt_ext="png", image_ext="jpg"))
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=0,
            **meta,
        )

    

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_cs(_root)
print("cs registered !")
DatasetCatalog.get("cs_train")
