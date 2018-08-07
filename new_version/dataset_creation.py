from sklearn.datasets import make_regression
import numpy as np
import base
import importlib
importlib.reload(base)

n_samples = 200
seed = 50  #50 is cool


class LinearDataset(base.BaseDataset):
    def __init__(self, n_samples=n_samples, seed=seed):
        super(LinearDataset, self).__init__(n_samples=n_samples, seed=seed)

    def create_dataset(self):
        X, y = make_regression(n_samples=n_samples, n_features=1, noise=15,
                               random_state=self.seed,
                               shuffle=True)  #,n_informative=1,bias=100)
        y = y / 10  #+ 10
        return X, y


class NonlinearDataset(base.BaseDataset):
    def __init__(self, n_samples=n_samples, seed=seed,
                 generating_function=None):
        self.generating_function = generating_function or self.base_generating_function
        super(NonlinearDataset, self).__init__(n_samples=n_samples, seed=seed)

    def base_generating_function(self, X):
        return np.sin(X).ravel() * 10 + np.random.normal(
            0, 1, size=X.shape[0])  # + 10

    def create_dataset(self):

        # Create a random dataset
        rng = np.random.RandomState(self.seed)
        X = np.sort(9 * rng.rand(n_samples, 1), axis=0)
        y = self.generating_function(X)

        return X, y
