"""
评估指标：准确率、精确率、召回率、F1、混淆矩阵
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix as sk_confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> dict:
    """
    计算分类评估指标。

    Args:
        y_true:  (N,) 真实标签
        y_pred:  (N,) 预测标签
        average: 多分类平均方式: 'macro', 'micro', 'weighted'

    Returns:
        dict: {"accuracy", "precision", "recall", "f1"}
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """返回混淆矩阵 (num_classes, num_classes)。"""
    return sk_confusion_matrix(y_true, y_pred)


def compute_metrics_per_class(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """
    计算每个类别的精确率、召回率、F1。
    论文第5章表5.3/5.4对每个模型给出了这四项指标。
    """
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
    }
