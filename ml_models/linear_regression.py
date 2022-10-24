import numpy as np

class BaseRegression:

    def __init__(self):

        self._fitted = False

    def fit(self, X: np.array, y: np.array):
        try:
            assert len(X) == len(y)
        except AssertionError:
            print('Number of samples in X and y should be equal! Fitting aborted.')
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

        return np.matmul(X, self.w)


class LinearRegression(BaseRegression):

    def _fit(self, X: np.array, y: np.array):
        self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), y)
        return self
        


class StochasticLinearRegression:
    
    def __init__(self):
        pass

    def fit(self, X: np.array, y: np.array):
        pass

    def predict(self, X: np.array) -> np.array:
        pass

# L1 regularization
class LassoRegression(BaseRegression):
    
    def __init__(self, alpha=0.01):
        super.__init__()
        self.alpha = alpha

    def _fit(self, X: np.array, y: np.array):
        pass


# L2 regularization
class RidgeRegression:
    
    def __init__(self, alpha=0.01):
        super.__init__()
        self.alpha = alpha

    def _fit(self, X: np.array, y: np.array):
        
        self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X) + self.alpha*np.eye(len(X))), np.transpose(X)), y)

        return self
