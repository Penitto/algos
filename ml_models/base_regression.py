import numpy as np


class BaseRegression:

    def __init__(self):

        self._fitted = False

    def fit(self, X: np.array, y: np.array):
        try:
            assert len(X) == len(y)
        except AssertionError:
            print('Number of samples in X and y should be equal! \
                Fitting aborted.')
            return

        self._fit(X, y)
        self._fitted = True

        return self

    def predict(self, X: np.array) -> np.array:
        try:
            assert self._fitted
        except AssertionError:
            print('Model is not fitted yet!')
            return

        try:
            assert X.shape[1] == len(self.w)
        except AssertionError:
            print('Dimensions don\'t match! Impossible to predict.')
            return

        return self._predict(X)
