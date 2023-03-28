  
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import random
# fmt: off
import sys
from test_utils import compute_fscore, compute_iou
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
from custom_model import CustomModel
custom = CustomModel()
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from cnn_model import CNN
from mask_former import add_mask_former_config
from predictor import VisualizationDemo

#python .\demo\test_AIS.py --config-file configs\ade20k-150\maskformer_R101_bs16_160k.yaml --input 
#datasets\cs\train\images\2.jpg --opts MODEL.WEIGHTS models\model_final_1aeb94.pkl
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

        x_dir = os.path.join('datasets\\ais_test\\images')
        y_dir = os.path.join('datasets\\ais_test\\annotations')
        images = os.listdir(x_dir)
        annotations = os.listdir(y_dir)
        print(images)
        print(annotations)

        iou_mean = 0
        fscore = 0
        inf = 0
        cnn = CNN()
        count = 0
        for i,z in zip(images,annotations):
            img = cv2.imread(x_dir+ "\\" + i)
            print(img.shape)
           
            
            label = cv2.imread(y_dir + "\\"+ z, cv2.IMREAD_GRAYSCALE)
            print(label.shape)
            label[label>0] = 255
            t1= time.perf_counter()
            p= cnn.detectImg(img)
            p = int(p)
          
            count = count +1
            if p == 0 :
                print("more person")
                predictions, visualized_output = demo.run_on_image(img)

            
                # visualized_output[300:480, 400:580]=0
                pred = visualized_output
         
                # print(predictions)
                iden = []
                for i in predictions['panoptic_seg'][1]:
                    if i['category_id'] == 12:
                        iden.append(i['id'])
        
                vis =visualized_output.get_image()[:, :, ::-1]
                s = predictions['panoptic_seg'][0].numpy()
            
               
                s= s.astype(np.uint8)

                for x in iden:
                    s[s==x] = 255
                s[s != 255] = 0
                
      
                # cv2.imshow('demo', s)
                # cv2.imshow('vis', vis)
                # while True:
                #     k = cv2.waitKey(50) & 0xFF
                #     if k == ord('q'):
                #         break
                iou_mean = iou_mean + compute_iou(s, label)
                fscore = fscore + compute_fscore(s, label)

                
              
          
            else:
                print("one person")
            # img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                predictions = custom.detectImg(img)
                # print(np.unique(predictions))
                print(np.unique(predictions))


                # print(np.unique(label))
            
           
                iou_mean = iou_mean + compute_iou(predictions, label)
                fscore = fscore + compute_fscore(predictions, label)
                cv2.imshow("demo", predictions)

                # while True:
                #     k = cv2.waitKey(50) & 0xFF
                #     if k == ord('q'):
                #         break
            t2 = time.perf_counter()
            inf = inf + (t2 - t1)
           

    print("iou is", iou_mean/8)
    print("fscore is", fscore / 8)
    print(f"inference is {inf/8:.10f} seconds")

        