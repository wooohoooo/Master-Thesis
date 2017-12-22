import numpy as np
from plot import plot_prediction


def rmse(y_hat, y):
    return np.sqrt(np.mean((y_hat - y)**2))


def evaluate_model(X, y, y_hat, var=None, plot=False, sorted_index=None,
                   generating_func=None):
    print('RSME is {}'.format(rmse(y_hat, y)))
    #if plot:
    #    plot_prediction(y, y_hat, sorted_index, lr_var,
    #                    generating_func=generating_func)
