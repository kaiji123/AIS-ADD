
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from PIL import Image
import numpy as np
import cv2
import segmentation_models as sm
import tensorflow as tf
import os 


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import cv2
import mss
import numpy as np
import tensorflow as tf

sm.set_framework('tf.keras')

class CustomModel:

    def __init__(self) -> None:
        
        dice_loss = sm.losses.DiceLoss( beta = 1.5) 
        total_loss = dice_loss + 1.5 * sm.losses.JaccardLoss()
        BACKBONE = 'efficientnetb3'

        CLASSES = ['player']
        LR = 0.0008
        n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
        activation = 'sigmoid' if n_classes == 1 else 'softmax'
        self.model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

        optim = tf.keras.optimizers.Adam(LR)
        
        metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

        # compile keras model with defined optimozer, loss and metrics
        self.model.compile(optim, total_loss, metrics)
        # Load the model
        self.model.load_weights("models\\unet.h5") 



    def detectImg(self,img):
        
        # img = np.array(sct.grab({"top": top, "left": left, "width": width, "height": height}))
        
    
        image = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # Convert the image to a numpy array
        image_array =cv2.resize(image, (320,320), interpolation= cv2.INTER_LINEAR)
        # print(image_array.shape)
        # Add an extra dimension to the image (since Keras expects a batch of images)
        image_array = np.expand_dims(image_array, axis=0)

        t1 = time.perf_counter()
        # Predict the class of the image
        predictions = self.model.predict(image_array).round()

        t2 = time.perf_counter()
        elapsed_time = t2 - t1

        # Print the elapsed time
    
        # print(f"Inference took {elapsed_time:.10f} seconds")

        predictions =predictions.squeeze()

        predictions = cv2.resize(predictions,(642,480),interpolation=cv2.INTER_NEAREST)

        predictions[predictions == 1] = 255
        cv2.imshow("demo", predictions)
        cv2.waitKey(1)
  









