import matplotlib.pyplot as plt
import numpy as np
from networks import CopyNetwork
from base import EnsembleNetwork
import scipy
import tensorflow as tf


class BootstrapEnsemble(object):
    def __init__(self, ensemble=None, num_features=None, num_epochs=10,
                 num_ensembles=5, seed=42, num_neurons=[10, 5, 3],
                 initialisation_scheme=None, activations=None):
        self.num_features = num_features or 1
        self.seed = seed
        self.num_neurons = num_neurons
        self.activations = activations

        self.ensemble_list = ensemble or [EnsembleNetwork] * num_ensembles
        self.num_epochs = num_epochs
        self.initialisation_scheme = initialisation_scheme or tf.contrib.layers.xavier_initializer
        self.initialise_ensemble()

    def initialise_ensemble(self):
        self.ensemble = [
            member(num_features=self.num_features, seed=i + self.seed,
                   num_epochs=self.num_epochs, num_neurons=self.num_neurons,
                   initialisation_scheme=self.initialisation_scheme,
                   activations=self.activations)
            for i, member in enumerate(self.ensemble_list)
        ]

    def fit(self, X, y):
        '''This is where we build in the Online Bootstrap'''
        #for i in range(self.num_epochs):

        for estimator in self.ensemble:
            estimator.fit(X, y)

    def get_prediction_list(self, X):
        pred_list = []
        for estimator in self.ensemble:
            prediction = estimator.predict(X)
            pred_list.append(prediction)
        return pred_list

    def predict(self, X):
        pred_list = self.get_prediction_list(X)
        predictive_mean = np.mean(pred_list, 0)

        return predictive_mean

    def predict_std(self, X):

        pred_list = self.get_prediction_list(X)
        predictive_uncertainty = np.std(pred_list, 0) or 0.001

        return predictive_uncertainty

    def get_mean_and_std(self, X):
        pred_list = self.get_prediction_list(X)
        pred_mean = np.mean(pred_list, axis=0)
        pred_std = np.std(pred_list, axis=0)
        return pred_mean, pred_std

    def compute_rsme(self, X, y):
        y_hat = self.predict(X)
        return np.sqrt(np.mean((y_hat - y)**2))

    def compute_error_vec(self, X, y):
        y_hat = self.predict(X)
        return y - y_hat

    def compute_nlpd(self, X, y):
        y_hat, std = self.get_mean_and_std(X, std=True)

        return -1 / 2 * np.mean(
            safe_ln(std) + ((y_hat - y)**2 / (std + 0.0001)))

    def compute_coverage_probability(self, X, y):

        y_hat, std = self.get_mean_and_std(X, std=True)
        #print(y_hat.shape,std.shape,y.shape)

        CP = 0
        for pred, s, target in zip(y_hat, std, y):
            #print(len(pred))
            #print(len(s))
            #print(len(target))
            if pred + s > target > pred - s:
                CP += 1
        return CP / len(y)

    def compute_CoBEAU(self, X, y):
        prediction, variance = self.get_mean_and_std(X, std=True)

        error = (prediction - y)**2
        correlation = scipy.stats.pearsonr(error.flatten(), variance.flatten())

        #np.correlate(error.flatten(),variance.flatten())
        return correlation


class BootstrapThroughTimeBobStrap(BootstrapEnsemble):
    #TODO: decide if replace every epoch or meta-epoch
    #TODO: Early stopping if error does not decrease

    def __init__(self, num_features=None, num_epochs=10, num_models=3,
                 model_name='copynetwork', seed=42, num_neurons=[10, 5, 3],
                 initialisation_scheme=None, activations=None):
        self.model_name = 'checkpoints/' + model_name
        self.activations = activations
        self.model = CopyNetwork(
            seed=seed, initialisation_scheme=initialisation_scheme,
            activations=activations, num_neurons=num_neurons,
            num_epochs=num_epochs)
        self.train_iteration = 0

        super(BootstrapThroughTimeBobStrap, self).__init__(
            ensemble=None, num_features=num_features, num_epochs=num_epochs,
            num_ensembles=1, seed=seed, num_neurons=num_neurons,
            initialisation_scheme=initialisation_scheme)
        self.num_epochs = num_epochs
        self.num_models = num_models

    def initialise_ensemble(self):
        """create list of checkpoint ensembles"""
        name = self.model_name + 'checkpoint_' + str(self.train_iteration)
        self.model.save(name)
        self.checkpoints = [name]

    def get_prediction_list(self, X):
        prediction_list = []
        for ckpt in self.checkpoints:
            self.model.load(ckpt)
            prediction_list.append(self.model.predict(X))

        return prediction_list

    def fit(self, X, y, X_test=None, y_test=None):
        """trains the most recent model in checkpoint list and replaces the oldest checkpoint if enough checkpoints exist"""
        for i in range(self.num_epochs):
            self.train_iteration += 1
            name = self.model_name + '_checkpoint_' + str(self.train_iteration)

            self.model.load(self.checkpoints[-1])  #load most recent model
            self.model.fit(X, y)  #train most recent model
            self.model.save(name)  #save newest model as checkpoint
            self.checkpoints.append(name)  #add newest checkpoint

            if len(
                    self.checkpoints
            ) > self.num_models:  #if we reached max number of stored models
                self.checkpoints.pop(0)  #delete oldest checkpoint


class ForcedDiversityBootstrapThroughTime(BootstrapThroughTimeBobStrap):
    # TODO: try out 'burn in' Phase.
    def __init__(self, num_features=None, num_epochs=1, num_models=10,
                 model_name='diversitycopynetwork', seed=42,
                 num_neurons=[10, 5, 3], initialisation_scheme=None,
                 activations=None):

        super(ForcedDiversityBootstrapThroughTime, self).__init__(
            num_features=None, num_epochs=num_epochs, num_models=num_models,
            model_name='forceddiversitycopynetwork', seed=seed,
            num_neurons=num_neurons,
            initialisation_scheme=initialisation_scheme,
            activations=activations)

    def fit(self, X, y, X_test=None, y_test=None):
        """trains the most recent model in checkpoint list and replaces the oldest checkpoint if enough checkpoints exist"""
        for i in range(self.num_epochs):
            self.train_iteration += 1
            name = self.model_name + '_checkpoint_' + str(self.train_iteration)

            self.model.load(self.checkpoints[-1])  #load most recent model
            rsme_before = self.model.compute_rsme(X, y)
            self.model.fit(X, y)  #train most recent model
            rsme_after = self.model.compute_rsme(X, y)
            if rsme_before > rsme_after:
                self.model.save(name)  #save newest model as checkpoint
                self.checkpoints.append(name)  #add newest checkpoint

                if len(
                        self.checkpoints
                ) > self.num_models:  #if we reached max number of stored models
                    self.checkpoints.pop(0)  #delete oldest checkpoint


class ForcedDiversityBootstrapThroughTime2(BootstrapThroughTimeBobStrap):
    def __init__(self, num_features=None, num_epochs=1, num_models=10,
                 model_name='diversitycopynetwork', seed=42,
                 num_neurons=[10, 5, 3], initialisation_scheme=None,
                 activations=None):

        super(ForcedDiversityBootstrapThroughTime2, self).__init__(
            num_features=None, num_epochs=num_epochs, num_models=num_models,
            model_name='forceddiversitycopynetwork', seed=seed,
            num_neurons=num_neurons,
            initialisation_scheme=initialisation_scheme,
            activations=activations)

    def fit(self, X, y, X_test=None, y_test=None):
        """trains the most recent model in checkpoint list and replaces the oldest checkpoint if enough checkpoints exist"""
        for i in range(self.num_epochs):
            self.train_iteration += 1
            name = self.model_name + '_checkpoint_' + str(self.train_iteration)

            self.model.load(self.checkpoints[-1])  #load most recent model
            error_vec_before = self.compute_error_vec(X, y)**2
            self.model.fit(X, y)  #train most recent model
            error_vec_after = self.model.compute_error_vec(X, y)**2

            #maybe use self.compute_error!!!!

            kl_divergence = scipy.stats.entropy(error_vec_before,
                                                error_vec_after)
            print(kl_divergence)

            if np.round(kl_divergence, decimals=1) <= 0:
                continue

            self.model.save(name)  #save newest model as checkpoint
            self.checkpoints.append(name)  #add newest checkpoint

            if len(
                    self.checkpoints
            ) > self.num_models:  #if we reached max number of stored models
                self.checkpoints.pop(0)  #delete oldest checkpoint


class ForcedDiversityBootstrapThroughTime3(
        ForcedDiversityBootstrapThroughTime):
    """implements only getting the std from the ensemble, pred_mean is just last prediction"""

    def __init__(self, num_features=None, num_epochs=1, num_models=10,
                 model_name='diversitycopynetwork', seed=42,
                 num_neurons=[10, 5, 3], initialisation_scheme=None,
                 activations=None):

        super(ForcedDiversityBootstrapThroughTime3, self).__init__(
            num_features=None, num_epochs=num_epochs, num_models=num_models,
            model_name='forceddiversitycopynetwork', seed=seed,
            num_neurons=num_neurons,
            initialisation_scheme=initialisation_scheme,
            activations=activations)

    def predict(self, X):
        self.model.load(self.checkpoints[-1])
        return self.model.predict(X)
        #return predictive_uncertainty

    def get_mean_and_std(self, X):
        pred_list = self.get_prediction_list(X)
        self.model.load(self.checkpoints[-1])
        pred_mean = self.model.predict(X)
        #prediction_list.append(self.model.predict(X))
        pred_std = np.std(pred_list, axis=0)
        return pred_mean, pred_std