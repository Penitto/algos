from .base_clustering import BaseClustering
import numpy as np


class KMeans(BaseClustering):

    def __init__(self,
                 k: int,
                 scaling: str,
                 max_iter: int,
                 tol: float,
                 distance_func):

        super().__init__(k, scaling, distance_func)
        self.max_iter = max_iter
        self.tol = tol

    def _fit(self, X: np.array):

        self.anchors = np.random.uniform(
            low=self.col_min,
            high=self.col_max,
            size=(self.k, self.X.shape[1]))

        for _ in range(self.max_iter):

            distances = self.distance_func(self.anchors, self.X)
            classes = np.argmin(distances, axis=1)

            self.anchors = np.array(
                [
                    distances[classes == i].mean(axis=0)
                    for i in range(self.k)])

    def _predict(self, X: np.array):
        distances = self.distance_func(self.anchors, X)
        classes = np.argmin(distances, axis=1)

        return classes
