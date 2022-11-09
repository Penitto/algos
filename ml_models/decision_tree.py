import numpy as np
from collections import Counter


class DecisionTree:

    @staticmethod
    def mse_imputiry(y: np.array):
        mean = np.mean(y)
        return 1 / len(y) * np.sum(np.square(y - mean))

    @staticmethod
    def mae_inpurity(y: np.array):
        median = np.median(y)
        return 1 / len(y) * np.sum(np.abs(y - median))

    @staticmethod
    def entropy_impurity(y: np.array):
        _, c = np.unique(y, return_counts=True)
        freq = c / len(y)
        return -np.sum(freq * np.log(freq))

    @staticmethod
    def gini_impurity(y: np.array):
        _, c = np.unique(y, return_counts=True)
        freq = c / len(y)
        return np.sum(freq*(1-freq))

    @staticmethod
    def get_splits(col: np.array):
        unique = np.unique(col)
        return np.array([unique[:-1], unique[1:]]).mean(axis=0)

    def __init__(self,
                 task: str = None,
                 max_depth: int = 5,
                 split_method: str = 'gini'):

        try:
            assert task in ['classification', 'regression']
        except AssertionError:
            print('Please specify task')
            return

        try:
            assert max_depth > 0
        except AssertionError:
            print('Max depth should be positive')
            return

        try:
            assert split_method in ['gini', 'entropy', 'mae', 'mse']
        except AssertionError:
            print('Uknown split method')
            return

        self._methods = {
            'gini': self.gini_impurity,
            'entropy': self.entropy_impurity,
            'mse': self.mse_imputiry,
            'mae': self.mae_inpurity
        }

        self._split_method = split_method
        self._task = task
        self._nodes = {}
        self._max_depth = max_depth
        self._fitted = False

    def find_best_split(self, feature_vector, target_vector):

        splits = self.get_splits(feature_vector)
        HR = self._methods[self._split_method]
        branch = []
        for i in splits:
            left = feature_vector < i
            right = feature_vector >= i
            branch.append(
                len(feature_vector) * HR(target_vector) -
                np.sum(left) * HR(target_vector[left]) -
                np.sum(right) * HR(target_vector[right]))

        return splits[np.argmax(branch)], branch[np.argmax(branch)]

    def _fit_node(self, X: np.array, y: np.array, node: dict):

        # if all labels are the same
        if np.all(y == y[0]):
            node["type"] = "terminal"
            node["class"] = y[0]
            return

        feature_best, threshold_best, criterion_best, split = None, None, None, None
        for feature in range(X.shape[1]):
            feature_vector = X[:, feature]

            threshold, criterion = self.find_best_split(feature_vector, y)

            # Если нашёлся признак с критерием лучше, чем был
            if criterion_best is None or criterion > criterion_best:
                feature_best = feature
                criterion_best = criterion
                split = feature_vector < threshold
                threshold_best = threshold

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        node["threshold"] = threshold_best
        # И пошли строить дерево дальше
        # с условием, что у нас будет, что отправить и налево и направо
        node["left"], node["right"] = {}, {}
        self._fit_node(X=X[split], y=y[split], node=node['left'])
        self._fit_node(X=X[~split], y=y[~split], node=node['right'])

    def fit(self, X: np.array, y: np.array):
        self._fit_node(X=X, y=y, node=self._nodes)

    def predict(self, X: np.array) -> np.array:
        
        try:
            assert self._fitted
        except AssertionError:
            print('Model is not fitted')
            return

        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
