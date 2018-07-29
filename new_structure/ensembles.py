import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = [12, 4]
import scipy

import base
import importlib
importlib.reload(
    base
)  #this is for notebooks - otherwise I have to restart the kernel every time I change anything

from networks import EnsembleNetwork  #, GaussianLearningRateEstimator, GaussianLossEstimator


class VanillaEnsemble(base.BasePredictor):
    """
	https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/05_Ensemble_Learning.ipynb
	"""

    def __init__(self, estimator_stats=None, num_epochs=10):

        default_ensemble = [{
            'num_neurons': [2, 2, 2],
            'num_epochs': num_epochs
        }, {
            'num_neurons': [10, 2],
            'num_epochs': num_epochs
        }, {
            'num_neurons': [5, 5],
            'num_epochs': num_epochs
        }]

        self.estimator_stats = estimator_stats or default_ensemble
        self.estimator_list = [
            EnsembleNetwork(**x) for x in self.estimator_stats
        ]

    def fit(self, X, y):
        '''This is where we build in the Online Bootstrap'''

        X = self.check_input_dimensions(X)
        y = self.check_input_dimensions(y)
        for estimator in self.estimator_list:
            #estimator.train_and_evaluate(X, y,shuffle=False)
            estimator.train(X, y)
        #system('say training  complete')

    def predict(self, X, return_samples=False):
        pred_list = []
        for estimator in self.estimator_list:
            prediction = estimator.predict(X)
            pred_list.append(prediction)
        #for i,sample in enumerate(pred_list):
        #   assert(np.isnan(sample) is False), 'sample {} contains NaN'.format(i)
        stds = np.std(pred_list, 0)
        means = np.mean(pred_list, 0)
        #assert(np.isnan(stds) is False)
        #assert(np.isnan(means) is False)
        return_dict = {'stds': stds, 'means': means}
        if return_samples:
            return_dict['samples'] = pred_list
        #system('say prediction complete')

        return return_dict


class BootstrapEnsemble(base.BasePredictor):
    """
	https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/05_Ensemble_Learning.ipynb
	"""

    def __init__(self, estimator_stats=None, num_estimators=10, num_epochs=10,
                 seed=10, learning_rate=0.05):

        default_stats = {'num_neurons': [10, 10], 'num_epochs': num_epochs}

        self.num_estimators = num_estimators
        self.estimator_stats = estimator_stats or default_stats
        est_list = [EnsembleNetwork] * self.num_estimators
        self.estimator_list = [x(**self.estimator_stats) for x in est_list]
        self.dataset_list = None

    def fit(self, X, y):
        '''This is where we build in the Online Bootstrap'''

        X = self.check_input_dimensions(X)
        y = self.check_input_dimensions(y)
        if not self.dataset_list:
            self.dataset_list = []
            old_mask = None
            for i in range(self.num_estimators):
                np.random.seed = i

                #WTF
                mask = np.random.randint(0, 30, size=len(X)) > 15
                assert (mask is not old_mask)
                dataset = np.array(X)[mask]
                labels = np.array(y)[mask]
                #print(len(dataset))
                self.dataset_list.append({'X': dataset, 'y': labels})
                old_mask = mask

        for estimator, dataset in zip(self.estimator_list, self.dataset_list):
            estimator.train(dataset['X'], dataset['y'], shuffle=False)
        #system('say training complete')

    def predict(self, X, return_samples=False):
        pred_list = []
        for estimator in self.estimator_list:
            prediction = estimator.predict(X)
            pred_list.append(prediction)

        stds = np.std(pred_list, 0)
        means = np.mean(pred_list, 0)

        return_dict = {'stds': stds, 'means': means}
        if return_samples:
            return_dict['samples'] = pred_list
        #system('say prediction complete')

        return return_dict
