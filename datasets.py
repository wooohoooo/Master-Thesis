import numpy as np


def make_y(X, noise=True):
    func = 5 * np.sin(X) + 10
    if noise:
        noise = np.random.normal(0, 4, size=X.shape)
        return func + noise
    return func


def expand_array_dims(array):
    new_array = [np.expand_dims(np.array(x), 0) for x in array]
    #new_array = [np.expand_dims(x,1) for x in new_array]

    return new_array