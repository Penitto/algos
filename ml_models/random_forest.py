import numpy as np


class RandomForest:

    def __init__(self, task=None):

        assert task in ['classification', 'regression']
        self._task = task

    def fit(self, X: np.array, y: np.array):
        pass

    def predict(self, X: np.array) -> np.array:
        pass
