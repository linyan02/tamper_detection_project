import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def accuracy_score_new(y_true, y_pred):
    """计算分类准确率（正确预测数 / 总样本数）"""
    y_true = np.array(y_true).flatten()  # 转换为numpy数组并展平
    y_pred = np.array(y_pred).flatten()  # 确保维度一致
    return np.mean(y_true == y_pred)  # 逐元素比较，求平均值

def f1_score_new(y_true, y_pred):
    return f1_score(np.array(y_true), np.array(y_pred))

def iou_score(mask_true, mask_pred):
    intersection = np.logical_and(mask_true, mask_pred).sum()
    union = np.logical_or(mask_true, mask_pred).sum()
    return intersection / union if union > 0 else 0