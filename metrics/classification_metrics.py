import numpy as np


def true_positive(y_true: np.array, y_pred: np.array):
    return np.logical_and(y_true, y_pred).sum()


def true_negative(y_true: np.array, y_pred: np.array):
    return len(y_true) - np.logical_or(y_true, y_pred).sum()


def false_positive(y_true: np.array, y_pred: np.array):
    """
    a(x) = 1 but y = 0
    """
    return np.logical_and(np.logical_xor(y_true, y_pred), y_pred)


def false_negative(y_true: np.array, y_pred: np.array):
    """
    a(x) = 0 but y = 1
    """
    return np.logical_and(np.logical_xor(y_true, y_pred), y_true)


def accuracy(y_true: np.array, y_pred: np.array) -> float:
    return np.sum(y_true == y_pred) / len(y_true)


def precision(y_true: np.array, y_pred: np.array) -> float:
    tp = true_positive(y_true=y_true, y_pred=y_pred)
    fp = false_positive(y_true=y_true, y_pred=y_pred)
    return tp / (tp + fp)


def recall(y_true: np.array, y_pred: np.array) -> float:
    tp = true_positive(y_true=y_true, y_pred=y_pred)
    fn = false_negative(y_true=y_true, y_pred=y_pred)
    return tp / (tp + fn)
