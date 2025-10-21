import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def accuracy_score(y_true, y_pred):
    return accuracy_score(np.array(y_true), np.array(y_pred))

def f1_score(y_true, y_pred):
    return f1_score(np.array(y_true), np.array(y_pred))

def iou_score(mask_true, mask_pred):
    intersection = np.logical_and(mask_true, mask_pred).sum()
    union = np.logical_or(mask_true, mask_pred).sum()
    return intersection / union if union > 0 else 0