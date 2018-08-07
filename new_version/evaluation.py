import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import pandas as pd


def safe_ln(x):
    if x < 0:
        return 0
    else:
        return np.ln(x)


def safe_ln(x):
    return np.log(x + 0.0001)


def safe_ln(x):
    x[x < 0] = 10000
    logs = np.log(x)
    logs[x < 0] = 0
    return logs


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


def repeat_experiment(model_creator, dataset_creator, num_meta_epochs=2,
                      plot=True, model_params={}, datset_params={}):
    meta_start_time = time.time()

    pred_list = []
    cobeau_list = []
    nlpd_list = []
    coverage_list = []
    rsme_list = []
    dataset = dataset_creator(**datset_params)  #create a dataset
    X_train, y_train = dataset.train_dataset  #get dataset

    X_test, y_test = dataset.test_dataset  #get test data

    test_idx = dataset.return_test_idx
    train_idx = dataset.return_train_idx

    time_list = []

    for i in range(num_meta_epochs):
        start_time = time.time()
        model_params['seed'] = i + 42

        #model = model_creator(**model_params)  #create a model
        model = model_creator(**model_params)  #create a model
        #dataset = dataset_creator(**datset_params)  #create a dataset

        model.fit(X_train, y_train)  #,shuffle=True)
        y_pred, y_var = model.get_mean_and_std(X_test)
        pred_list.append({
            'prediction': y_pred,
            'X_test': X_test,
            'y_test': y_test
        })

        cobeau_list.append(compute_CoBEAU(y_pred, y_test, y_var))
        nlpd_list.append(compute_nlpd(y_pred, y_test, y_var))
        coverage_list.append(
            compute_coverage_probability(y_pred, y_test, y_var))
        rsme_list.append(compute_rsme(y_pred, y_test))

        time_exp = time.time() - start_time
        time_list.append(time_exp)
        if i % 10 == 0:
            print(
                'experiment number {} took {} seconds. That means the whole run will probably take {} more seconds and {} more minutes'.
                format(  #'1', '2', '3', '4'))
                    str(i + 1),
                    str(time_exp),
                    str(np.mean(time_list) * (num_meta_epochs - i)),
                    str(np.mean(time_list) * (num_meta_epochs - i) / 60)))
    if plot:

        plt.figure()
        plt.plot(cobeau_list)
        plt.title('cobeau')
        plt.xlabel('experiment')
        plt.ylabel('cobeau')

        plt.figure()
        plt.plot(nlpd_list)
        plt.title('nlpd')
        plt.xlabel('experiment')
        plt.ylabel('nlpd')

        plt.figure()
        plt.plot(coverage_list)
        plt.title('coverage')
        plt.xlabel('experiment')
        plt.ylabel('coverage')

        plt.figure()
        plt.plot(rsme_list)
        plt.title('rsme')
        plt.xlabel('experiment')
        plt.ylabel('rsme')

    print('overall, it took {} seconds with {} experiments'.format(
        time.time() - meta_start_time, num_meta_epochs))

    value_dict = {
        'nlpd': nlpd_list,
        'rsme': rsme_list,
        'cobeau': [entry[1] for entry in cobeau_list],
        'coverage': coverage_list
    }
    try:
        print(pd.DataFrame.from_records(value_dict).describe())
        print(pd.DataFrame.from_records(value_dict).describe().to_latex())
        return pd.DataFrame.from_records(value_dict)
    except:
        return value_dict