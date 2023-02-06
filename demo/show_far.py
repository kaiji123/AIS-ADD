# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

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

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask_former import add_mask_former_config
from predictor_test import VisualizationDemo

#python demo\demo.py --config-file 'C:\Users\Kai Ji\Desktop\Maskformer\MaskFormer\configs\myconfig.yaml' --input 'C:\Users\Kai Ji\Desktop\Maskformer\MaskFormer\datasets\cs\test\images\36.jpg' --opts MODEL.WEIGHTS output\model_final.pth
# constants
WINDOW_NAME = "MaskFormer demo"


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


from sklearn.metrics import confusion_matrix 
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
def compute_iou(y_pred, y_true, plot=False):
    actual = np.random.binomial(1,.9,size = 1000)
    print(actual.shape)
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    print(np.unique(y_pred))
    y_true = y_true.flatten()
    print(np.unique(y_true))
    labels = [0, 1]
    current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(current)

    
    cm = current
    print(cm)
    if plot == True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    print(current)
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)

    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    print("iou",iou_score)
    mean = np.mean(IoU)
    return iou_score
def compute_fscore(predicted, ground_truth, plot=False, beta = 1):
    # print(np.unique(predicted))
    # print(np.unique(ground_truth))
    true_positives = np.sum(np.logical_and(predicted == 1, ground_truth == 1))
    false_positives = np.sum(np.logical_and(predicted == 1, ground_truth == 0))
    false_negatives = np.sum(np.logical_and(predicted == 0, ground_truth == 1))
    # print(true_positives)
    # print(false_negatives)
    # print(false_positives)
    # precision = true_positives / (true_positives + false_positives)
    # recall = true_positives / (true_positives + false_negatives)

    # # Calculate the F-score
    # Fscore = 2 * (precision * recall) / (precision + recall)
    Fscore = ((1 + beta**2) * true_positives) / ((1+ beta ** 2)* true_positives + beta**2 * false_negatives + false_positives)

    return Fscore

from keras import backend as K
# def iou_coef(y_true, y_pred, smooth=1):
#   intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
#   union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
#   iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
#   return iou

# def dice_coef2(y_true, y_pred, smooth = 1):
#     y_pred = y_pred.flatten()
#     y_true = y_true.flatten()
#     np.sum(y_pred)
#     dice = 2*intersection / union.astype(np.float32)
#     return dice



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


    if args.input:
        # if len(args.input) == 1:
        x_root = 'C:\\Users\\Kai Ji\\Desktop\\Maskformer\\MaskFormer\\datasets\\far'
        x_files = os.listdir(x_root)
        for i in x_files:
          
            
            x= cv2.imread(x_root + "\\" + i)
            print(x.shape)
            cv2.imshow('x',x)
            # image1 = cv2.rectangle(x, (400,300), (600,480), (0,0,0), -1) 
            # print(image1.shape)
            # cv2.imshow('region',image1)
            x = cv2.cvtColor(x, cv2.COLOR_RGBA2BGR)
            print(np.unique(x))
            predictions, visualized_output = demo.run_on_image(x)
            visualized_output[visualized_output==1] = 255
            print(np.unique(predictions))
         
            print(np.unique(visualized_output))
            cv2.imshow('pred',visualized_output)

            if cv2.waitKey(0) == 27:
                pass

    
