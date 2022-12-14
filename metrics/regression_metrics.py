import numpy as np


def mse(y_true: np.array, y_pred: np.array) -> float:
    ans = np.matmul(np.transpose(y_true - y_pred), y_true - y_pred).sum()
    ans /= len(y_true)
    return ans


def mae(y_true: np.array, y_pred: np.array) -> float:
    ans = np.abs(y_true - y_pred).sum() / len(y_true)
    return ans


def rmse(y_true: np.array, y_pred: np.array) -> float:
    return np.sqrt(mse(y_true=y_true, y_pred=y_pred))


def r2(y_true: np.array, y_pred: np.array) -> float:
    mean = np.sum(y_true) / len(y_true)
    ans = 1 - mse(y_true=y_true, y_pred=y_pred) / \
        mse(y_true=y_true, y_pred=np.full_like(y_true, mean))
    return ans


def msle(y_true: np.array, y_pred: np.array) -> float:
    ans = mse(y_true=np.log(y_true + 1), y_pred=np.log(y_pred + 1))
    return ans


def mape(y_true: np.array, y_pred: np.array) -> float:
    ans = np.abs(y_true - y_pred)
    ans /= np.abs(y_true)
    ans = ans.sum() / len(y_true)

    return ans
