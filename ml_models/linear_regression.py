from .base_regression import BaseRegression
import numpy as np


class BaseLinearRegression(BaseRegression):

    def _predict(self, X: np.array):

        return X @ self.w


class LinearRegression(BaseLinearRegression):

    def _fit(self, X: np.array, y: np.array):
        self.w = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
        return self


class SGDLinearRegression(BaseLinearRegression):

    def __init__(self, eta: float):
        
        super().__init__()
        self.eta = eta

    def fit(self, X: np.array, y: np.array):
        pass

    def predict(self, X: np.array) -> np.array:
        pass


# L2 regularization
class RidgeLinearRegression(BaseLinearRegression):

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def _fit(self, X: np.array, y: np.array):

        self.w = np.linalg.inv(
            np.transpose(X) @ X + self.alpha * np.eye(len(X))
        ) @ np.transpose(X) @ y
        return self
