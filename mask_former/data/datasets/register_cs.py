# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from sys import platform
<<<<<<< HEAD
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
=======
COCO_CATEGORIES = [
    {  "isthing": 1, "id": 1, "name": "person"},
    { "isthing": 0, "id": 0, "name": "notperson"}
]


def _get_coco_stuff_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing.
    stuff_ids = [k["id"] for k in COCO_CATEGORIES]
    assert len(stuff_ids) == 2, len(stuff_ids)
>>>>>>> origin/test

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
<<<<<<< HEAD
    meta = _get_meta()
=======
    meta = _get_coco_stuff_meta()
>>>>>>> origin/test
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
<<<<<<< HEAD
            ignore_label=0,
=======
            ignore_label=65535,
>>>>>>> origin/test
            **meta,
        )

    

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_cs(_root)
<<<<<<< HEAD
print("cs registered !")
=======
>>>>>>> origin/test
DatasetCatalog.get("cs_train")
