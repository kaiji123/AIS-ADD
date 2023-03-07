# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py

import sys
# sys.path.append('yolov5')
sys.path.append('yolo_new')
import argparse
import glob
import multiprocessing as mp
import os
import random
# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on
# import register_cs
import tempfile
import time
import warnings
from sys import platform
if platform == 'win32':
    import cv2
import numpy as np
import tqdm
import mss
# from yolov5.aimbot import YoloModel
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from yolo_new.aimbot import YoloModel
from mask_former import add_mask_former_config
from predictor import VisualizationDemo
from custom_model import CustomModel
from cnn_model import CNN ,AlexNet, AlexFar, CNNFar
#python demo\demo.py --config-file 'C:\Users\Kai Ji\Desktop\Maskformer\MaskFormer\configs\myconfig.yaml' --input 'C:\Users\Kai Ji\Desktop\Maskformer\MaskFormer\datasets\cs\test\images\36.jpg' --opts MODEL.WEIGHTS output\model_final.pth
# constants

from test_utils import compute_iou, compute_fscore
WINDOW_NAME = "MaskFormer demo"
custom = CustomModel()

cnn = CNNFar()
obj_det = YoloModel()
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/ade20k-150/maskformer_R50_bs16_160k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    cfg.defrost()
    cfg.MODEL.DEVICE = 'cpu'
    demo = VisualizationDemo(cfg)
    
    offset = 30
    times = []
    sct = mss.mss()
        
    top = 30
    left = 0
    width =640
    height =480
    autoaim = False

    x_dir = os.path.join('datasets\\cs_test\\test\\images')
    y_dir = os.path.join('datasets\\cs_test\\test\\annotations')
    images = os.listdir(x_dir)
    annotations = os.listdir(y_dir)
    print(images)


    iou_mean = 0
    fscore = 0
    for i,z in zip(images,annotations):
        img = cv2.imread(x_dir+ "\\" + i)
        print(img.shape)
        label = cv2.imread(y_dir + "\\"+ z, cv2.IMREAD_GRAYSCALE)
        print(label.shape)
        cnn.detectImg(img)
        label[label > 0] = 255
        cv2.imshow("label", label)
        s = random.choice([0,1])
        if s == 1 :

            img = cv2.resize(img, (640,640), interpolation= cv2.INTER_LINEAR)
            image , boxes, coords = obj_det.predict_coords(img)
            # print(np.unique(image))
            print(coords)
            cv2.imshow("origin", boxes)
            boxes = cv2.cvtColor(boxes, cv2.COLOR_BGR2GRAY)
            boxes[:,:]= 0
            
            for inst in coords:
                
                x1 = int(inst[0].item())
                y1 = int(inst[1].item())
                x2 = int(inst[2].item())
                y2 = int(inst[3].item())
                boxes[y1:y2, x1:x2] = 255
            boxes[boxes> 0] = 255

            
            
         
            cv2.imshow("demo", boxes)

            predictions = cv2.resize(boxes,(642,480),interpolation=cv2.INTER_LINEAR)
            

            while True:
                k = cv2.waitKey(50) & 0xFF
                if k == ord('q'):
                    break
            print(boxes)
            iou_mean = iou_mean + compute_iou(predictions, label)
            fscore = fscore + compute_fscore(predictions, label)
            
            
        else:
            # img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            predictions = custom.detectImg(img)
            # print(np.unique(predictions))
            predictions[predictions > 0 ]= 255
            # print(np.unique(label))
          
            cv2.imshow("demo", predictions)
            iou_mean = iou_mean + compute_iou(predictions, label)
            fscore = fscore + compute_fscore(predictions, label)
            while True:
                k = cv2.waitKey(50) & 0xFF
                if k == ord('q'):
                    break


    print("iou is", iou_mean/8)
    print("fscore is", fscore / 8)

