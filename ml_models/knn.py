from .base_clustering import BaseClustering
import numpy as np


class KNNClassifier(BaseClustering):

    def _fit(self, X: np.array, y: np.array):
        return self

    def _predict(self, X: np.array):

        classes = self.y[np.argsort(
                    self.distance_func(
                        self.X, X)
                    )[:, :self.k]]

        classes, counts = np.unique(classes, return_counts=True, axis=1)

        res = classes[counts.argmax(axis=1)]

        return res


class KNNRegression(BaseClustering):

    def _fit(self, X: np.array, y: np.array):
        return self

    def _predict(self, X: np.array):

        res = np.mean(
                np.argsort(
                    self.distance_func(
                        self.X, X)
                    )[:, :self.k], axis=1)

        return res
