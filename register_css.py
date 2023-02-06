# # Copyright (c) Facebook, Inc. and its affiliates.
# import os

# from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets import load_sem_seg
# from sys import platform
# COCO_CATEGORIES = [
#     {  "isthing": 1, "id": 1, "name": "person"},
#     # { "isthing": 0, "id": 0, "name": "notperson"}
# ]


# def _get_coco_stuff_meta():
#     # Id 0 is reserved for ignore_label, we change ignore_label for 0
#     # to 255 in our pre-processing.
#     stuff_ids = [k["id"] for k in COCO_CATEGORIES]
#     assert len(stuff_ids) == 2, len(stuff_ids)

#     # For semantic segmentation, this mapping maps from contiguous stuff id
#     # (in [0, 91], used in models) to ids in the dataset (used for processing results)
#     stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
#     stuff_classes = [k["name"] for k in COCO_CATEGORIES]

#     ret = {
#         "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
#         "stuff_classes": stuff_classes,
#     }
#     return ret


# def register_cs(root):
#     print(root)
#     root = os.path.join(root, "cs")
#     meta = _get_coco_stuff_meta()
#     directory= [
#         ("train", "train/images", "train/annotations"),
#         ("test", "test/images", "test/annotations"),
#     ]
#     if platform == 'win32':
#         directory = [
#         ("train", "train\\images", "train\\annotations"),
#         ("test", "test\\images", "test\\annotations"),
#     ]
#     for name, image_dirname, sem_seg_dirname in directory:
#         image_dir = os.path.join(root, image_dirname)
#         gt_dir = os.path.join(root, sem_seg_dirname)
#         name = f"cs_{name}"
#         print("dict: ",load_sem_seg(gt_dir, image_dir, gt_ext="png", image_ext="jpg"))
#         DatasetCatalog.register(
#             name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
#         )
#         MetadataCatalog.get(name).set(
#             image_root=image_dir,
#             sem_seg_root=gt_dir,
#             evaluator_type="sem_seg",
#             ignore_label=65535,
#             **meta,
#         )


# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
# register_cs(_root)
# DatasetCatalog.get("cs_train")
