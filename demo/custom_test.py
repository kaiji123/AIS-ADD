from sklearn.metrics import confusion_matrix
from tensorflow import keras
from PIL import Image
import numpy as np
import cv2
import segmentation_models as sm
import efficientnet.keras 
from efficientnet.tfkeras import EfficientNetB4
import tensorflow as tf
import os 
import matplotlib.pyplot as plt

dice_loss = sm.losses.DiceLoss( beta = 1.5) 
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + 1.5 * sm.losses.JaccardLoss()

import segmentation_models as sm
sm.set_framework('tf.keras')
# segmentation_models could also use `tf.keras` if you do not have Keras installed
# or you could switch to other framework using `sm.set_framework('tf.keras')`

BACKBONE = 'efficientnetb3'
BATCH_SIZE = 8
CLASSES = ['player']
LR = 0.0008
EPOCHS = 80

preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'



def compute_iou(y_pred, y_true, plot=False):
    # actual = np.random.binomial(1,.9,size = 1000)
    # print(actual.shape)
    # # ytrue, ypred is a flatten vector
    # y_pred = y_pred.flatten()
    # print(np.unique(y_pred))
    # y_true = y_true.flatten()
    # print(np.unique(y_true))
    # labels = [0, 1]
    # current = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # print(current)

    
    # cm = current
    # print(cm)
    # if plot == True:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     cax = ax.matshow(cm)
    #     plt.title('Confusion matrix of the classifier')
    #     fig.colorbar(cax)
    #     ax.set_xticklabels([''] + labels)
    #     ax.set_yticklabels([''] + labels)
    #     plt.xlabel('Predicted')
    #     plt.ylabel('True')
    #     plt.show()
    # print(current)
    # # compute mean iou
    # intersection = np.diag(current)
    # ground_truth_set = current.sum(axis=1)
    # predicted_set = current.sum(axis=0)
    # union = ground_truth_set + predicted_set - intersection
    # IoU = intersection / union.astype(np.float32)
    # print("iou",iou_score)
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
 
    # mean = np.mean(IoU)
    return iou_score
def compute_fscore(predicted, ground_truth, plot=False, beta = 1):
    # print(predicted.shape)
    # print(ground_truth.shape)
    
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
#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

optim = tf.keras.optimizers.Adam(LR)

# # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# # bce_loss = sm.losses.BinaryCELoss()
# focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
# total_loss = iou_loss + (1 * focal_loss)
# # total_loss = bce_loss + sm.losses.DiceLoss()
dice_loss = sm.losses.DiceLoss( beta = 1.5) 
focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + 1.5 * sm.losses.JaccardLoss()

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)
# Load the model
model.load_weights("models\\unet.h5") 

s =os.listdir('datasets\\far\\test\\images')
# Load the image and resize it to the input shape of the model
# print(s)

y_pred =[]
y_true = []
count =0
for i in s:
    # print(i)
# image = cv2.imread("datasets\\cs\\test\\images\\36.jpg")
    image = cv2.imread("datasets\\far\\test\\images\\" + i)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



    # Convert the image to a numpy array
    image_array =cv2.resize(image, (320,320), interpolation= cv2.INTER_LINEAR)
    print(image_array.shape)
    img= image_array.copy()
    # Add an extra dimension to the image (since Keras expects a batch of images)
    image_array = np.expand_dims(image_array, axis=0)


    # Predict the class of the image
    predictions = model.predict(image_array).round()
    print(predictions.shape)
    print("unique", np.unique(predictions))
    predictions =predictions.squeeze()
    print(np.unique(predictions))
    predictions = cv2.resize(predictions,(642,480),interpolation=cv2.INTER_NEAREST)
    print("unique",np.unique(predictions))
    # cv2.imshow('demo unet',predictions[0])
    # cv2.imshow('image',img)
    print(predictions.shape)
  
    y_pred.append(predictions)
    # cv2.imshow("pred",predictions)
    num = i.split('.')
    num = num[0]
    y = cv2.imread("datasets\\far\\test\\annotations\\" + num + ".png",0)

    print(np.unique(y))
    
    y[y<=128] =0
    y[y>128] = 1
    # y[y==1] = 255
    print(y.shape)

    # cv2.imshow('truth',y)
    y_true.append(y)

    # if cv2.waitKey(0) == 27:
    #     print("quitting")
print(len(y_pred))
y_pred = np.array(y_pred)
y_true = np.array(y_true)
print("the mean is ", compute_iou(y_pred=y_pred, y_true=y_true))
print("f1-score is", compute_fscore( y_pred, y_true))