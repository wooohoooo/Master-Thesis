from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import numpy as np
import base
import importlib
importlib.reload(base)

num_samples = 200
seed = 50  #50 is cool


class LinearDataset(base.BaseDataset):
    def __init__(self, num_samples=num_samples, seed=seed):
        super(LinearDataset, self).__init__(num_samples=num_samples, seed=seed)

    def create_dataset(self, scale=False):
        X, y = make_regression(n_samples=self.num_samples, n_features=1,
                               noise=15, random_state=self.seed,
                               shuffle=True)  #,n_informative=1,bias=100)
        y = y / num_samples  #10  #+ 10
        if scale:
            y = (y - min(y)) / (max(y) - min(y))
            X = (X - min(X)) / (max(X) - min(X))

        return X, y


class NonlinearDataset(base.BaseDataset):
    def __init__(self, num_samples=num_samples, seed=seed,
                 generating_function=None, scope=9):  #sope=4.5
        self.generating_function = generating_function or self.base_generating_function
        self.scope = scope
        self.num_samples = num_samples

        super(NonlinearDataset, self).__init__(num_samples=self.num_samples,
                                               seed=seed)

    def base_generating_function(self, X):
        return np.sin(X).ravel() * 1 + np.random.normal(
            0, .1, size=X.shape[0])  # + 10

    def make_X(self):
        rng = np.random.RandomState(self.seed)

        X = np.sort(self.scope * rng.rand(self.num_samples, 1), axis=0) - (
            self.scope / 2)
        return X

    def create_dataset(self, scale=False):

        # Create a random dataset
        X = self.make_X()
        y = self.generating_function(X)
        if scale:
            y = (y - min(y)) / (max(y) - min(y))
            X = (X - min(X)) / (max(X) - min(X))

        return X, y


class XThreeDataset(NonlinearDataset):
    def base_generating_function(self, X):
        y = X.T**3 + np.random.normal(0, 3, size=X.shape[0])
        y = y.T
        return y.flatten()
