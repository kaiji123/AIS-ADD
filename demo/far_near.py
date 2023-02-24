# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py

import sys
sys.path.append('yolov5')
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
from yolov5.aimbot import YoloModel
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask_former import add_mask_former_config
from predictor import VisualizationDemo
from custom_model import CustomModel
from cnn_model import CNN ,AlexNet, AlexFar, CNNFar
#python demo\demo.py --config-file 'C:\Users\Kai Ji\Desktop\Maskformer\MaskFormer\configs\myconfig.yaml' --input 'C:\Users\Kai Ji\Desktop\Maskformer\MaskFormer\datasets\cs\test\images\36.jpg' --opts MODEL.WEIGHTS output\model_final.pth
# constants
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


    if args.input:
        # if len(args.input) == 1:

        while True:
            t1 = time.perf_counter()
            # print("hello")
            img = np.array(sct.grab({"top": top, "left": left, "width": width, "height": height}))
            predcnn = cnn.detectImg(img)
            predcnn = int(predcnn)

            talex = time.perf_counter()

            elaps_alex = talex - t1
            print(f"alexnet took {elaps_alex:.10f} seconds")
            # print(predcnn)
            
            if predcnn == 1:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                img = cv2.resize(img, (640,640), interpolation= cv2.INTER_LINEAR)
                image , boxes = obj_det.predict(img)
                cv2.imshow("demo", image)
                
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                
                predictions, visualized_output = demo.run_on_image(img)
                vis =visualized_output.get_image()[:, :, ::-1]
                # print("hello")
                cv2.imshow('demo',vis)
            
            t2 = time.perf_counter()
            elapsed_time = t2 - t1

            # Print the elapsed time
        
            # print(f"Inference took {elapsed_time:.10f} seconds")
            # args.input = glob.glob(os.path.expanduser(args.input[0]))
            # assert args.input, "The input path(s) was not found"
            # for path in tqdm.tqdm(args.input, disable=not args.output):
            #     # use PIL, to be consistent with evaluation
            # img = read_image(path, format="BGR")

            
            # start = time.perf_counter()
            # predictions, visualized_output = demo.run_on_image(img)
            # end = time.perf_counter()
            # print("inference time",  end - start)
            # logger.info(
            #     "{}: {} in {:.2f}s".format(
            #         path,
            #         "detected {} instances".format(len(predictions["instances"]))
            #         if "instances" in predictions
            #         else "finished",
            #         time.time() - start_time,
            #     )
            # )
            # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            # vis =visualized_output.get_image()[:, :, ::-1]
            # print("hello")
            # cv2.imshow('WINDOW_NAME',vis)
            # if cv2.waitKey(25) & 0xFF == ord("p"):
            #     cv2.destroyAllWindows()
            #     break

            cv2.waitKey(1)
