import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split


def generate_sinoid(X):
    return 5 * np.sin(X) + 10 + X**2


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


def unison_shuffled_copies(a, b, expand_dims=False):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    sorted_index = np.argsort(p)
    if expand_dims == True:
        return expand_array_dims(a[p]), expand_array_dims(b[p]), sorted_index
    else:
        p = np.squeeze(p)
        return a[p], b[p], sorted_index


def make_dataset(seed=None, x_start=-5, x_end=5, sample_size=100,
                 generating=generate_sinoid, train_p=None):
    if seed:
        np.random.seed(seed)
    x_data = np.linspace(x_start, x_end, sample_size)
    y_true = make_y(x_data, generating=generate_sinoid)
    #np.random.seed(103)
    #if shuffled:
    x_data_shuffled, y_true_shuffled, sorted_index = unison_shuffled_copies(
        x_data, y_true, expand_dims=True)

    dataset = {
        'X': x_data_shuffled,
        'y': y_true_shuffled,
        'shuffle_index': sorted_index,
        'generating': generating
    }

    if type(train_p) == float:
        until = int(sample_size * train_p)

        X_train = x_data_shuffled[:until]
        y_train = y_true_shuffled[:until]
        train_index = sorted_index[:until]

        X_test = x_data_shuffled[until:]
        y_test = y_true_shuffled[until:]
        test_index = sorted_index[until:]

        dataset['X_train'] = X_train
        dataset['y_train'] = y_train
        dataset['train_index'] = np.argsort(X_train, axis=0)

        dataset['X_test'] = X_test
        dataset['y_test'] = y_test
        dataset['test_index'] = np.argsort(X_test, axis=0)

    return dataset


def make_cross_validation_dataset(seed=42, x_start=-5, x_end=5,
                                  sample_size=100, test_size=0.3,
                                  generating=generate_sinoid):

    data = make_dataset(seed, x_start, x_end,
                        sample_size + sample_size * test_size, generate_sinoid)

    X_train, X_test, y_train, y_test = train_test_split(
        data['X'], data['y'], test_size=test_size, random_state=seed)

    data['X_train'] = expand_array_dims(X_train)
    data['y_train'] = expand_array_dims(y_train)
    data['X_test'] = expand_array_dims(X_test)
    data['y_test'] = expand_array_dims(y_test)
    data['train_shuffle'] = np.argsort(data['X_train'])
    assert (len(data['train_shuffle']) is len(X_train))
    data['test_shuffle'] = np.argsort(data['X_test'])
    return data
