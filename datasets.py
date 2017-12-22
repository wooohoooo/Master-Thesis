import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def generate_sinoid(X):
    return 5 * np.sin(X) + 10


def make_y(X, noise=True, generating=generate_sinoid):
    func = generating(X)
    if noise:
        noise = np.random.normal(0, 4, size=X.shape)
        return func + noise
    return func


def expand_array_dims(array):
    new_array = [np.expand_dims(np.array(x), 0) for x in array]
    #new_array = [np.expand_dims(x,1) for x in new_array]

    return new_array


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    sorted_index = np.argsort(p)
    return expand_array_dims(a[p]), expand_array_dims(b[p]), sorted_index


def make_dataset(seed=None, x_start=-5, x_end=5, sample_size=100,
                 generating=generate_sinoid):
    if seed:
        np.random.seed(seed)
    x_data = np.linspace(x_start, x_end, sample_size)
    y_true = make_y(x_data, generating=generate_sinoid)
    #np.random.seed(103)
    #if shuffled:
    x_data_shuffled, y_true_shuffled, sorted_index = unison_shuffled_copies(
        x_data, y_true)

    dataset = {
        'X': x_data_shuffled,
        'y': y_true_shuffled,
        'shuffle_index': sorted_index,
        'generating': generating
    }
    return dataset
