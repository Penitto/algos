from .base_regression import BaseRegression
import numpy as np
import random


class BaseLinearRegression(BaseRegression):

    def _predict(self, X: np.array):

        return X @ self.w


class LinearRegression(BaseLinearRegression):

    def _fit(self, X: np.array, y: np.array):
        self.w = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
        return self


class SGDLinearRegression(BaseLinearRegression):

    def __init__(self, eta: float, max_iter: int):

        super().__init__()
        self.eta = eta
        self.max_iter = max_iter

    def _fit(self, X: np.array, y: np.array):

        self.w = np.random.randn((X.shape[1], 1))

        for _ in range(self.max_iter):

            # amount of features to use
            amount = random.randint(0, len(X))

            # stochastic means not for random features
            feat = np.random.choice(len(self.X), amount, replace=False)

            grad = 2 / amount * (X[feat] @ self.w - y[feat]).T @ X[feat]

            self.w = self.w - self.eta * grad

        return self


# L2 regularization
class RidgeLinearRegression(BaseLinearRegression):

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def _fit(self, X: np.array, y: np.array):

        self.w = np.linalg.inv(
            X.T @ X + self.alpha * np.eye(len(X))
        ) @ X.T @ y
        return self
