import numpy as np
import scipy


def safe_ln(x):
    if x < 0:
        return 0
    else:
        return np.ln(x)


def safe_ln(x):
    return np.log(x + 0.0001)


def compute_rsme(pred, y_test):
    return np.mean((pred - y_test)**2)


def compute_nlpd(y_hat, y, std):
    #y_hat, std = self.get_mean_and_std(X, std=True)
    nlpd = -1 / 2 * np.mean(safe_ln(std) + ((y_hat - y)**2 / (std + 0.0001)))
    if np.isnan(nlpd):
        return 0

    return nlpd


def compute_coverage_probability(y_hat, y, std):

    #y_hat, std = self.get_mean_and_std(X, std=True)
    #print(y_hat.shape,std.shape,y.shape)

    CP = 0
    for pred, s, target in zip(y_hat, std, y):
        #print(len(pred))
        #print(len(s))
        #print(len(target))
        if pred + s > target > pred - s:
            CP += 1
    return CP / len(y)


def compute_CoBEAU(prediction, y, variance):
    #prediction, variance = self.get_mean_and_std(X, std=True)

    error = (prediction - y)**2
    correlation = scipy.stats.pearsonr(error.flatten(), variance.flatten())

    #np.correlate(error.flatten(),variance.flatten())
    return correlation