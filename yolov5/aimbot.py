#!/usr/bin/env python3.8
import os

import torch

from models.common import DetectMultiBackend

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import mss
import numpy as np
from prediction import predict
mx= 0
my= 0
mw = 0
mh = 0

# coords = getWindowsCoords() # get window coords
# print(coords)
# x,y,w,h = coords
sct = mss.mss()
# pyautogui settings
import pyautogui # https://totalcsgo.com/commands/mouse
pyautogui.MINIMUM_DURATION = 0
pyautogui.MINIMUM_SLEEP = 0
pyautogui.PAUSE = 0

offset = 30
times = []
sct = mss.mss()








# top = 30
# left = 0
# width =640
# height =480
# autoaim = False
class YoloModel:
    def __init__(self):
        
        model = DetectMultiBackend('yolov5\\runs\\train\\yolov5s_results8\\weights\\best.pt', device=torch.device('cpu'), dnn=False, data='yolov5\dataset\data.yaml', fp16=False)
        self.model = model
    def predict(self, image):
        return predict(source=image, model= self.model)

# while True:
    
#     img = np.array(sct.grab({"top": top, "left": left, "width": width, "height": height}))
#     # cx,cy = getCenter(left, top, left + width, top + height)
#     img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
#     t1 = time.perf_counter()
#     image, boxes= predict(source=img, model=model)
    
#     t2 = time.perf_counter()
#     elapsed_time = t2 - t1

#     # Print the elapsed time
  
#     # print(f"Inference took {elapsed_time:.10f} seconds")
#     times.append(t2-t1)
#     times = times[-50:]
#     ms = sum(times)/len(times)*1000
#     fps = 1000 / ms
#     # print("FPS", fps)
#     # image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
#     cv2.imshow('test',image)
#     # print(boxes)

   
#     # if autoaim == True:
#     #     if len(boxes) > 0:
#     #         index = chooseEDistance(boxes, cx, cy)
#     #         x1,y1,x2,y2 = boxes[index]
#     #         bx , by = getCenter(x1,y1,x2,y2)
#     #         x, y = calculateDistance(cx,cy,bx,by)
#     #         pyautogui.move(x,y)
        
   
    
#     # th_list, t_list = [], []
#     # for detection in detection_list:
#     #     diff_x = (int(w/2) - int(detection[1]))*-1
#     #     diff_y = (int(h/2) - int(detection[2]))*-1
#     #     if detection[0] == "th":
#     #         th_list += [diff_x, diff_y]
#     #     elif detection[0] == "t":
#     #         t_list += [diff_x, diff_y]

#     # if len(th_list)>0:
#     #     new = min(th_list[::2], key=abs)
#     #     index = th_list.index(new)
#     #     pyautogui.move(th_list[index], th_list[index+1])
#     #     if abs(th_list[index])<12:
#     #         pyautogui.click()
#     # elif len(t_list)>0:
#     #     new = min(t_list[::2], key=abs)
#     #     index = t_list.index(new)
#     #     pyautogui.move(t_list[index], t_list[index+1])
#     #     if abs(t_list[index])<12:
#     #         pyautogui.click()

    
    
  
#     if cv2.waitKey(25) & 0xFF == ord("p"):
#         cv2.destroyAllWindows()
#         break

#     cv2.waitKey(1)
# cv2.destroyAllWindows()