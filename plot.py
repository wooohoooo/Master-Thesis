import matplotlib.pyplot as plt
import numpy as np


def plot_dataset(X, y, sorted_index, generating_func=None, legend=True):
    #X = [x[0] for x in X][sorted_index]
    #y = [y[0] for y in Y][sorted_index]
    X = np.squeeze(np.array(X))[sorted_index]
    y = np.squeeze(np.array(y))[sorted_index]

    plt.plot(X, y, '*', label='Dataset')
    if generating_func:
        y_generated = generating_func(X)
        plt.plot(X, y_generated, label='generating_func')
    if legend:
        plt.legend()


def plot_prediction(X, y, sorted_index, variance=None, generating_func=None,
                    legend=True):
    if type(variance) == np.ndarray:
        X = np.squeeze(np.array(X))[sorted_index]
        y = np.squeeze(np.array(y))[sorted_index]
        var = np.squeeze(np.array(variance))[sorted_index]

        plt.plot(X, y, label='prediction')
        plt.fill_between(X, y, y + var, alpha=.3, color='b',
                         label='uncertainty')
        plt.fill_between(X, y, y - var, alpha=.3, color='b')
        if generating_func:
            y_generated = generating_func(X)
            plt.plot(X, y_generated, label='generating_func')
        if legend:
            plt.legend()
    else:
        plot_dataset(X, y, sorted_index, generating_func=generating_func)


from evaluation import evaluate_model


def train_and_plot(model, X, y, sorted_index, num_eps=10, num_plots=5,
                   generating_func=None):
    error_list = []
    for i in range(num_eps):
        errors = model.train_and_evaluate(X, y)
        error_list += errors
        #gauss_lr.train(X,y)
        #vanilla.train(X,y)
        if i % (num_eps / num_plots) == 0:
            gauss_preds = model.predict(X)
            gauss_var = model.predict_var(X)

            #3lr_preds = gauss_lr.predict(X)
            #lr_var = gauss_lr.predict_var(X)

            #vanilla_preds = vanilla.predict(X)
            plot_prediction(X, gauss_preds, sorted_index, gauss_var,
                            generating_func=generating_func)
            plt.plot(X, y, 'x')
            plt.show()
            evaluate_model(X, y, gauss_preds, var=gauss_var)
    #plt.plot(np.squeeze(gauss_lr_error_list))
    return error_list


def plot_error(error_list):
    plt.plot(np.squeeze(error_list))


def plot_variance_prediction():
    plt.plot(x_data, preds_weird_at)
    plt.fill_between(x_data,
                     np.squeeze(preds_weird_at),
                     np.squeeze(preds_weird_at) + np.squeeze(variance),
                     alpha=.3, color='b')
    plt.fill_between(x_data,
                     np.squeeze(preds_weird_at),
                     np.squeeze(preds_weird_at) - np.squeeze(variance),
                     alpha=.3, color='b')
    plt.plot(x_data, y_true, 'x', label='true data', color='red')
    plt.plot(x_data, make_y(x_data, False), label='generating function')
    plt.title('new net')