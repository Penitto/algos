from .base_regression import BaseRegression
import numpy as np
import random


# Stochastic
class LogisticRegression(BaseRegression):

    @staticmethod
    def sigmoid(x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

    def __init__(self, eta: float, max_iter: int):
        super().__init__()
        self.eta = eta
        self.max_iter = max_iter

    def _fit(self, X: np.array, y: np.array):

        try:
            assert len(y.unique) == 2
        except AssertionError:
            print('This class for binary case only')
            return

        self.w = np.random.randn((X.shape[1], 1))

        for _ in range(self.max_iter):

            # amount of features to use
            amount = random.randint(0, len(X))

            # stochastic means not for random features
            feat = np.random.choice(len(self.X), amount, replace=False)

            grad = X[feat].T * (self.sigmoid(X[feat] @ self.w) - y[feat])

            self.w = self.w - self.eta * grad

        return self

    def predict_proba(self, X: np.array) -> np.array:
        return self.sigmoid(X @ self.w)

    def predict(self, X: np.array, cutoff: float) -> np.array:

        try:
            assert cutoff < 1 and cutoff > 0
        except AssertionError:
            print('Cutoff should be in (0; 1)')
            return

        probs = self.predict_proba(X)

        return np.where(probs >= cutoff, 1, 0)
