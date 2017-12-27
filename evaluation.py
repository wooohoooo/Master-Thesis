import numpy as np
from plot import plot_prediction


def rmse(y_hat, y):
    return np.sqrt(np.mean((y_hat - y)**2))


def coverage_probability(prediction, variance, truth):
    CP = 0
    for y_hat, s, y in zip(prediction, variance, truth):
        if y_hat + s > y > y_hat - s:
            CP += 1

    #if (y - 2 * s < 0) and (y + 2 * s > 0):
    #    CP += 1
    return CP / len(truth)


def evaluate_model(X, y, y_hat, var=None, plot=False, sorted_index=None,
                   generating_func=None):
    print('RSME is {}'.format(rmse(y_hat, y)))
    if type(var) is np.ndarray:
        cov_prob = coverage_probability(y_hat, var, y)
        print('COVERAGE PROBABILITY is {}'.format(cov_prob))
        print('COVERAGE/MEAN_VAR is {}'.format(cov_prob * 1.0 / np.mean(var)))
    #if plot:
    #    plot_prediction(y, y_hat, sorted_index, lr_var,
    #                    generating_func=generating_func)
