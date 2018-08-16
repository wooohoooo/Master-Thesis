import numpy as np
import scipy
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]

import time
import pandas as pd
from operator import itemgetter
from os import system
import datetime
import json
import pprint
import hashlib
pp = pprint.PrettyPrinter(indent=4)
from sklearn.preprocessing import StandardScaler


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
                      plot=True, model_params={}, dataset_params={}, seed=42):
    meta_start_time = time.time()
    print('experiment started at {}'.format(str(datetime.datetime.now())))

    pred_list = []
    cobeau_list = []
    nlpd_list = []
    coverage_list = []
    rsme_list = []
    dataset = dataset_creator(**dataset_params)  #create a dataset
    X_train, y_train = dataset.train_dataset  #get dataset

    X_test, y_test = dataset.test_dataset  #get test data

    test_idx = dataset.return_test_idx
    train_idx = dataset.return_train_idx

    time_list = []
    model_list = []

    for i in range(num_meta_epochs):
        start_time = time.time()
        model_params['seed'] = i + 42 + seed

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
        cobeau = compute_CoBEAU(y_pred, y_test, y_var)[1]
        cobeau_list.append(cobeau)
        nlpd = compute_nlpd(y_pred, y_test, y_var)
        nlpd_list.append(nlpd)
        cov = compute_coverage_probability(y_pred, y_test, y_var)
        coverage_list.append(cov)
        rsme = compute_rsme(y_pred, y_test)
        rsme_list.append(rsme)

        model_dict = {
            'prediction': y_pred,
            'uncertainty': y_var,
            'rsme': rsme,
            'nlpd': nlpd,
            'cov': cov,
            'cobeau': cobeau
        }
        model_list.append(model_dict)

        time_exp = time.time() - start_time
        time_list.append(time_exp)
        if i % 10 == 0:
            print(
                'experiment number {} took {} seconds. That means the whole run will probably take {} more seconds and {} more minutes.'.
                format(  #'1', '2', '3', '4'))
                    str(i + 1),
                    str(time_exp),
                    str(np.mean(time_list) * (num_meta_epochs - i)),
                    str(np.mean(time_list) * (num_meta_epochs - i) / 60)))
    if plot:

        plt.figure()
        plt.plot(cobeau_list, label='p')
        #plt.plot([entry[0] for entry in cobeau_list], label='r')
        plt.title('cobeau')
        plt.legend()
        plt.xlabel('experiment')
        plt.ylabel('cobeau')

        plt.figure()
        plt.plot(nlpd_list, label='nlpd')
        plt.title('nlpd')
        plt.xlabel('experiment')
        plt.ylabel('nlpd')
        plt.legend()

        plt.figure()
        plt.plot(coverage_list, label='coverage')
        plt.title('coverage')
        plt.xlabel('experiment')
        plt.ylabel('coverage')
        plt.legend()

        plt.figure()
        plt.plot(rsme_list, label='rmse')
        plt.title('rsme')
        plt.xlabel('experiment')
        plt.ylabel('rsme')
        plt.legend()

        #print best and worst model
        newlist = sorted(model_list, key=itemgetter('nlpd'), reverse=True)
        plt.figure()
        plt.scatter(X_test, y_test, label='test data')
        plt.scatter(X_train, y_train, label='train data')
        plt.plot(X_test[test_idx], newlist[0]['prediction'][test_idx],
                 label='best model')
        plt.fill_between(X_test[test_idx].ravel(),
                         newlist[0]['prediction'][test_idx].ravel(),
                         newlist[0]['prediction'][test_idx].ravel() -
                         newlist[0]['uncertainty'][test_idx].ravel(), alpha=.3,
                         color='b')
        plt.fill_between(X_test[test_idx].ravel(),
                         newlist[0]['prediction'][test_idx].ravel(),
                         newlist[0]['prediction'][test_idx].ravel() +
                         newlist[0]['uncertainty'][test_idx].ravel(), alpha=.3,
                         color='b')
        plt.title(
            'best model nlpd with a value of {}'.format(newlist[0]['nlpd']))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

        plt.figure()
        plt.scatter(X_test, y_test, label='test data')
        plt.scatter(X_train, y_train, label='train data')

        plt.plot(X_test[test_idx], newlist[-1]['prediction'][test_idx],
                 label='worst')
        plt.fill_between(X_test[test_idx].ravel(),
                         newlist[-1]['prediction'][test_idx].ravel(),
                         newlist[-1]['prediction'][test_idx].ravel() -
                         newlist[-1]['uncertainty'][test_idx].ravel(),
                         alpha=.3, color='b')
        plt.fill_between(X_test[test_idx].ravel(),
                         newlist[-1]['prediction'][test_idx].ravel(),
                         newlist[-1]['prediction'][test_idx].ravel() +
                         newlist[-1]['uncertainty'][test_idx].ravel(),
                         alpha=.3, color='b')
        plt.title('worst model nlpd with a value of {}'.format(
            model_list[-1]['nlpd']))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        #cobeau
        newlist = sorted(model_list, key=itemgetter('cobeau'), reverse=True)
        plt.figure()
        plt.scatter(X_test, y_test, label='test data')
        plt.scatter(X_train, y_train, label='train data')
        plt.plot(X_test[test_idx], newlist[0]['prediction'][test_idx],
                 label='best model')
        plt.fill_between(X_test[test_idx].ravel(),
                         newlist[0]['prediction'][test_idx].ravel(),
                         newlist[0]['prediction'][test_idx].ravel() -
                         newlist[0]['uncertainty'][test_idx].ravel(), alpha=.3,
                         color='b')
        plt.fill_between(X_test[test_idx].ravel(),
                         newlist[0]['prediction'][test_idx].ravel(),
                         newlist[0]['prediction'][test_idx].ravel() +
                         newlist[0]['uncertainty'][test_idx].ravel(), alpha=.3,
                         color='b')
        plt.title('best model cobeau with a value of {}'.format(
            newlist[0]['cobeau']))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

        plt.figure()
        plt.scatter(X_test, y_test, label='test data')
        plt.scatter(X_train, y_train, label='train data')
        plt.plot(X_test[test_idx], newlist[-1]['prediction'][test_idx],
                 label='worst')
        plt.fill_between(X_test[test_idx].ravel(),
                         newlist[-1]['prediction'][test_idx].ravel(),
                         newlist[-1]['prediction'][test_idx].ravel() -
                         newlist[-1]['uncertainty'][test_idx].ravel(),
                         alpha=.3, color='b')
        plt.fill_between(X_test[test_idx].ravel(),
                         newlist[-1]['prediction'][test_idx].ravel(),
                         newlist[-1]['prediction'][test_idx].ravel() +
                         newlist[-1]['uncertainty'][test_idx].ravel(),
                         alpha=.3, color='b')
        plt.title('worst model cobeau with a value of {}'.format(
            newlist[-1]['cobeau']))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

    print('overall, it took {} seconds with {} experiments'.format(
        time.time() - meta_start_time, num_meta_epochs))

    value_dict = {
        'nlpd': nlpd_list,
        'rsme': rsme_list,
        'cobeau': cobeau_list,
        'coverage': coverage_list
    }
    system('say Training and Evaluation has finished!')

    try:
        print(pd.DataFrame.from_records(value_dict).describe())
        print(pd.DataFrame.from_records(value_dict).describe().to_latex())
        return pd.DataFrame.from_records(value_dict)
    except:
        return value_dict


from sklearn.model_selection import ParameterGrid
import tensorflow as tf


#paraeter search
def gridsearch(model, dataset_creator, trials=10, seeds=[500, 1000, 1500],
               num_neurons=[[2, 3, 2], [10, 10, 10]],
               learning_rates=[0.1, 0.01, 0.001],
               activation_schemes=[tf.nn.leaky_relu, tf.sigmoid, tf.nn.tanh],
               initialisation_schemes=[
                   tf.keras.initializers.he_normal,
                   tf.contrib.layers.xavier_initializer
               ], l2=[False, True]):
    file_name = 'gridsearched_parameters/model_{}_{}'.format(
        str(type(model())).replace('.', '_').replace(' ', '').replace('<', '')
        .replace('>', '').replace("'", ""),
        str(type(dataset_creator())).replace('.', '_').replace(
            ' ', '').replace('<', '').replace('>', '').replace("'", ""))
    print(file_name)
    seed = 50
    ds = dataset_creator(seed=50)
    X_train, y_train = ds.train_dataset
    X_test, y_test = ds.test_dataset
    plt.scatter(X_train, y_train)
    plt.scatter(X_test, y_test)

    num_layers = len(num_neurons[0])
    activations = [[x] * num_layers for x in activation_schemes]

    param_grid = {
        'num_neurons': num_neurons,
        'activations': activations,
        'initialisation_scheme': initialisation_schemes,
        'learning_rate': learning_rates,
        'seed': seeds
    }

    grid = ParameterGrid(param_grid)
    print(len(grid))
    score_list = []
    time_list = []
    print('experiment started at {}. Doing {} trials each for {} combinations'.
          format(str(datetime.datetime.now()), str(trials), str(len(grid))))
    for i, params in enumerate(grid):
        start_time = time.time()
        scores = []
        for j in range(trials):
            net = model(**params)
            net.fit(X_train, y_train)
            scores.append(net.score(X_test, y_test))
        score = np.mean(scores)
        var = np.var(scores)
        score_list.append({'params': params, 'score': score, 'var': var})
        end_time = time.time()

        time_list.append(end_time - start_time)

        print(
            'took {} seconds ({} minutes) to do {} out of {}. Overall, estimated time is: {} minutes'.
            format(
                str(end_time - start_time),
                str((end_time - start_time) / 60),
                str(i),
                str(len(grid)), str((len(grid) - i) * np.mean(time_list) /
                                    10)))
    score_list = sorted(score_list, key=itemgetter('score'), reverse=False)
    pp.pprint(score_list[0])
    pp.pprint(score_list[-1])
    with open(file_name, 'w') as fout:
        json.dump(str(score_list), fout)
    return score_list


class ThompsonGridSearch(object):
    """takes a parameter grid, a meta-model 
    and a model for which the paramters are to be optimised 
    and performs thompson parameter search"""

    def __init__(self, param_grid, dataset_creator, thompson_model, test_model,
                 model_params):
        self.grid = ParameterGrid(param_grid)  #parameter grid
        self.scaler = StandardScaler()

        self.val_grid = self.create_dataset_from_grid()
        self.observed = []
        self.model_params = model_params

        self.thompson_model = thompson_model(
            num_features=self.input_size,
            **self.model_params)  #this is the model we're actually training
        self.test_model = test_model  #need to be initialiseable
        self.ds = dataset_creator()

    def create_input_from_params(self, params):
        pd_params = pd.from_records
        return np.array(list(params.items()))

    def create_dataset_from_grid(self):
        learning_rate_true = False
        df = pd.DataFrame(list(self.grid))
        if learning_rate_true == True:
            lrs = self.scaler.fit_transform(
                df['learning_rate'].values.reshape(-1, 1)).flatten()
            #print(lrs)
            df['learning_rates'] = lrs
        #df['learning_rate'] = df['learning_rate'].astype('str')

        df['activations'] = df['activations'].astype(str)
        df['initialisation_scheme'] = df['initialisation_scheme'].astype(str)
        df['num_neurons'] = df['num_neurons'].astype(str)
        df['seed'] = df['seed'].astype(str)
        df['l2'] = df['l2'].astype(str)
        df['optimizer'] = df['optimizer'].astype(str)
        df_dummies = pd.DataFrame(pd.get_dummies(df))
        vals = df_dummies.values
        self.input_size = vals.shape[1]
        return df_dummies.values

    def predict_grid(self):
        """predicts values for the whole grid"""
        predictions = {}
        for params in self.grid:
            """Check if this is already measured in which case add the measurement"""
            mean, var = self.thompson_model.get_mean_and_std(params)
            predictions[params] = {'mean': mean, 'var': var}
        return predictions

    def get_sample_grid(self):
        predictions = []
        for params, X in zip(self.grid, self.val_grid):
            """Check if this is already measured in which case add the measurement"""
            #print(X)
            X = self.thompson_model.check_input_dimensions(X)
            #print(X)
            mean, var = self.thompson_model.get_mean_and_std(np.transpose(X))
            var = np.sqrt(var)
            sample = self.sample_from_prediction(mean, var)
            predictions.append({
                'mean': mean,
                'var': var,
                'sample': sample,
                'params': params,
                'X': X
            })
        pred_sorted = sorted(
            predictions, key=itemgetter('sample'),
            reverse=False)  #or true? DO I minimise or maximise? RSME = Minimise
        return predictions

    def plot_sample_grid(self):
        predictions = self.get_sample_grid()
        X = [
            #str(abs(hash(str(prediction['params'])) % (10**8)))
            #'feature{}'.format(i) for i, prediction in enumerate(predictions)
            i for i, prediction in enumerate(predictions)
        ]
        y = [prediction['mean'] for prediction in predictions]
        y_var = [prediction['var'] for prediction in predictions]
        samples = [prediction['sample'] for prediction in predictions]
        #plt.figure()
        #plt.scatter(X, samples, color='r', label='sample')
        #plt.figure()
        #plt.scatter(X, y, color='b', label='mean')
        #plt.plot(X, samples, label='samples')
        #plt.plot(X, y, label='means')
        plt.scatter(X, y)
        plt.figure()
        plt.bar(X, y, alpha=0.4)
        plt.errorbar(X, y, yerr=y_var, alpha=0.4)

        #print(self.observed)
        param_obs = [str(observation['X']) for observation in self.observed]
        eXes = [str(observation['X']) for observation in predictions]
        #print('exes{}'.format(eXes))
        #print('params{}'.format(param_obs))

        X_observed = [
            eXes.index(np.array(observation)) for observation in param_obs
        ]

        y_observed = [observation['score'] for observation in self.observed]
        #print(eXes)
        #print(param_obs)
        plt.scatter(X_observed, y_observed, color='red', label='observations')
        plt.legend()
        plt.figure()
        plt.plot(X, samples, label='samples')
        plt.plot(X, y, label='means')
        #print(len(X))
        #print(len(y))
        #print(len(y_var))
        #X + y
        #y + y_var
        plt.fill_between(X, y,
                         np.array(y) + np.array(y_var), alpha=.3, color='b')
        plt.fill_between(X, y,
                         np.array(y) - np.array(y_var), alpha=.3, color='b')
        plt.scatter(X_observed, y_observed, color='red', label='observations')

        plt.legend()

        plt.figure()
        plt.scatter(X_observed, y_observed, label='observations')

        #print(self.observed)

    def observe(self, return_params=True):
        lr_scale = False
        predictions = self.get_sample_grid()
        pred_sorted = sorted(
            predictions, key=itemgetter('sample'),
            reverse=False)  #or true? DO I minimise or maximise? RSME = Minimise
        params = pred_sorted[-1]['params']

        params_as_data = pred_sorted[-1]['X']
        X_train, y_train = self.ds.train_dataset
        X_test, y_test = self.ds.test_dataset
        #print(params)
        #print(params_as_data)

        if lr_scale == True:
            params['learning_rate'] = self.scaler.inverse_transform(
                df['learning_rate'].values.reshape(-1, 1)).flatten()
        new_model = self.test_model(**params)
        new_model.fit(X_train, y_train)
        real_score = new_model.score(X_test, y_test)
        self.observed.append({
            'params': params,
            'score': real_score,
            'X': params_as_data
        })
        if return_params:
            #return np.expand_dims(np.array(real_score), 1), params_as_data
            return np.array(real_score), params_as_data
        return real_score

    def train_params(self, num_epochs=5, online_bootstrap=True):
        real_score, params = self.observe()
        #real_score = 
        #print(np.array([params.T]))
        #print(np.array(real_score))
        #print(np.array([params.T]).shape)
        #print(np.array(real_score).shape)
        X_new = np.array([np.array(params.T)])[0]
        y_new = np.expand_dims(np.array(real_score), 0)  #np.array(real_score)
        #print('huh')
        #print(X.shape)
        #print(y.shape)
        #print(X)
        #print(y)
        ##print(X)
        #print(y)

        #print(X.shape)

        #print(y.shape)

        #print(X)
        #print(y)
        if online_bootstrap:
            old_X = np.squeeze(
                np.array([x['X'] for x in self.observed]), axis=2)
            old_y = np.array([y['score'] for y in self.observed])
            num_points = len(old_y) + 1
            ps = [1 / (i + 1) for i in range(num_points)][::-1]
            mask = np.array(np.random.binomial(num_points, ps), dtype=bool)
            #print(mask)
            #print(y)
            #print(y[mask])
            #print('oldX{}'.format(old_X))
            #print(old_y)
            #print(old_X.shape)
            #print(X.shape)
            #print(old_y.shape)
            #print(y.shape)
            X = np.append(X_new.T, np.array(old_X).T, axis=1)
            y = np.append(y_new, np.array(old_y))
            #print(zip(X, y))
            X = X.T[mask].T
            y = y.T[:][mask].T
            #print(zip(X, y))
            #print(X.T[0])
            #print(X_new)
            #print(y[0])
            #print(y_new)
            print('length of the new dataset: {}'.format(X.shape))
            print('new X is in there: {}'.format(X.T in X_new))
            print('new y is in there: {}'.format(y in y_new))
            assert (X.T in X_new) == (y in y_new)
            print('mean probability is {}'.format(np.mean(mask)))
            #print(y[mask])
            #print(mask.shape)
            #print(y.shape)
            #print(X.shape)
        for i in range(num_epochs):

            #print(np.array([params.T]))
            #print(np.array(real_score))

            self.thompson_model.fit(X.T, y.T, shuffle=False)

    def goforit(self, num_times, num_epochs=1):
        for i in range(num_times):
            self.train_params(num_epochs)

    def sample_from_prediction(self, mean, var):
        return np.random.normal(loc=mean, scale=var, size=None)

    def get_best_observation(self):
        pred_sorted = sorted(self.observed, key=itemgetter('score'),
                             reverse=False)
        return pred_sorted[0], pred_sorted[-1]
        #def train_params_old(self, params):
        """trains a model on the params. Predicts X/y. Uses the outcome to train thompson_model."""

        #X_train, y_train = self.ds.train_dataset
        #X_test, y_test = self.ds.test_dataset
        #new_model = self.test_model(**params,).fit(X_train, y_train)
        #real_score = new_model.score(X_test, y_test)
        #self.thompson_model.fit(params, real_score)
