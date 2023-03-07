import numpy as np
def compute_iou(y_pred, y_true, plot=False):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def compute_fscore(y_pred, y_true, plot=False):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / (0.5 * (np.sum(union) - np.sum(intersection)) + np.sum(intersection))
    return iou_score