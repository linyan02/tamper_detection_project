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
    """Robust IoU: handle shapes, dtypes, probabilities and one-hot/multi-channel."""
    import numpy as np
    mt = np.asarray(mask_true)
    mp = np.asarray(mask_pred)

    # 如果是 one-hot / 多类别，取 argmax
    if mt.ndim == 3 and mt.shape[-1] > 1 and mp.ndim == 3 and mp.shape[-1] > 1:
        mt = mt.argmax(axis=-1)
        mp = mp.argmax(axis=-1)

    # 如果是 RGB 或 有通道的二值表示，合并为单通道
    if mt.ndim == 3 and mt.shape[-1] in (3,):
        mt = mt.sum(axis=-1) > 0
    if mp.ndim == 3 and mp.shape[-1] in (3,):
        mp = mp.sum(axis=-1) > 0

    # 如果概率或非布尔数组，阈值为 0.5 （根据需要调整）
    if mt.dtype != bool:
        mt = mt > 0.5
    if mp.dtype != bool:
        mp = mp > 0.5

    mt = mt.astype(bool)
    mp = mp.astype(bool)

    # 对形状仍不一致的情况做保护
    if mt.shape != mp.shape:
        raise ValueError(f"IoU: shape mismatch {mt.shape} vs {mp.shape}")

    inter = np.logical_and(mt, mp).sum()
    union = np.logical_or(mt, mp).sum()
    eps = 1e-6
    return float(inter) / (union + eps)
