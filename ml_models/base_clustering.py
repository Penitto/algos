import numpy as np


class BaseClustering:

    def __init__(self, k: int, scaling: str, distance_func):

        self.k = k

        try:
            assert scaling in ['min-max', 'norm']
        except AssertionError:
            print('Uknown scaling')
            return

        self._scaling = scaling

        # In case with train matrix shape of (rows1, dim)
        # and test matrix shape of (rows2, dim)
        # output should be (rows2, rows1)
        # input should be (train matrix, test matrix)
        self.distance_func = distance_func

        self._fitted = False

    def fit(self, X: np.array, y: np.array):
        self.X = X

        self.col_max = np.max(self.X, axis=0)
        self.col_min = np.min(self.X, axis=0)
        self.col_mean = np.mean(self.X, axis=0)
        self.col_std = np.std(self.X, axis=0)

        if self._scaling == 'min-max':
            self.X = (self.X - self.col_min) / \
                (self.col_max - self.col_min)
        else:
            self.X = (self.X - self.col_mean) / self.col_std

        self.y = y

        self._fitted = True
        return self

    def predict(self, X: np.array):

        try:
            assert self._fitted
        except AssertionError:
            print('Model is not fitted yet!')
            return

        if self._scaling == 'min-max':
            X = np.clip(
                (X - self.col_min) /
                (self.col_max - self.col_min),
                self._col_stats[1],
                self._col_stats[0])
        else:
            X = (X - self.col_mean) / self.col_std

        res = self._predict(X)

        return res
