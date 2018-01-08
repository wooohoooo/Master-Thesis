import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = [12, 4]

from estimators import EnsembleNetwork, GaussianLearningRateEstimator, GaussianLossEstimator


class EnsembleParent(object):
    """
    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/05_Ensemble_Learning.ipynb
    """

    def __init__(self, estimator_stats=None, num_epochs=10):
        pass

    def train(self, X, y):
        pass

    def predict(self, X, return_samples=False):
        pass

    def plot(self, X, y, plot_samples=False, original_func=None,
             sorted_index=None):
        preds_dict = self.predict(X, True)
        y_hats = np.array([mean[0] for mean in preds_dict['means']])
        y_stds = np.array([std[0] for std in preds_dict['stds']])
        samples = preds_dict['samples']
        X_plot = [x[0] for x in np.array(X)]  #[sorted_index]]

        #print(y_hats.shape)
        #print(y_stds.shape)
        #print(y.shape)
        #print(np.array(X).shape)
        #X_plot = X_plot[sorted_index]
        y = np.array(y)  #[sorted_index]
        y_hats = y_hats  #[sorted_index]
        y_stds = y_stds  #[sorted_index]

        plt.plot(X_plot, y, 'kx', label='Data', alpha=0.4)

        plt.plot(X_plot, y_hats, label='predictive mean')
        plt.fill_between(X_plot, y_hats + y_stds, y_hats, alpha=.3, color='b')
        plt.fill_between(X_plot, y_hats - y_stds, y_hats, alpha=.3, color='b')

        if plot_samples:
            for i, sample in enumerate(samples):

                plt.plot(X_plot, sample, label='sample {}'.format(i),
                         alpha=max(1 / len(samples), 0.3))
        if original_func:
            y_original = original_func(X_plot)
            plt.plot(X, y_original, label='generating model')
        plt.legend()


class VanillaEnsemble(EnsembleParent):
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

    def train(self, X, y):
        '''This is where we build in the Online Bootstrap'''
        for estimator in self.estimator_list:
            #estimator.train_and_evaluate(X, y,shuffle=False)
            estimator.train(X,y)
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


class BootstrapEnsemble(EnsembleParent):
    """
    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/05_Ensemble_Learning.ipynb
    """
    def __init__(self,estimator_stats = None,num_estimators=10,num_epochs=10,seed=10):
        
        default_stats = {'num_neurons':[10,10],'num_epochs':num_epochs}
        
        self.num_estimators = num_estimators
        self.estimator_stats = estimator_stats or default_stats
        est_list = [EnsembleNetwork] *self.num_estimators
        self.estimator_list = [x(**self.estimator_stats,
            seed=seed) for x in est_list]
        self.dataset_list = None
        
    def train(self,X,y):
        '''This is where we build in the Online Bootstrap'''
        if not self.dataset_list:
            self.dataset_list = []
            old_mask = None
            for i in range(self.num_estimators):
                np.random.seed=i

                #WTF
                mask = np.random.randint(0,30,size=len(X))>15
                assert(mask is not old_mask)
                dataset = np.array(X)[mask]
                labels = np.array(y)[mask]
                #print(len(dataset))
                self.dataset_list.append({'X':dataset,'y':labels})
                old_mask = mask
            

            
            
            
            
        for estimator,dataset in zip(self.estimator_list,self.dataset_list):
            estimator.train(dataset['X'],dataset['y'],shuffle=False)
        #system('say training complete')
            
            
    def predict(self,X,return_samples=False):
        pred_list = []
        for estimator in self.estimator_list:
            prediction = estimator.predict(X)
            pred_list.append(prediction)
            
        stds = np.std(pred_list,0)
        means = np.mean(pred_list,0)
        
        
        return_dict = {'stds':stds,'means':means}
        if return_samples:
            return_dict['samples'] = pred_list
        #system('say prediction complete')

        return return_dict
            


class GaussianNetworkEnsemble(EnsembleParent):
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
            GaussianLossEstimator(**x) for x in self.estimator_stats
        ]

    def train(self, X, y):
        '''This is where we build in the Online Bootstrap'''
        for estimator in self.estimator_list:
            estimator.train(X, y,online=True)
        #system('say training  complete')

    def predict(self, X, return_samples=False):
        pred_list = []
        var_list = []
        for estimator in self.estimator_list:
            prediction = estimator.predict(X)
            pred_list.append(prediction)
            var = estimator.predict_var(X)
            var_list.append(var)

        stds = np.mean(var_list, 0)
        means = np.mean(pred_list, 0)

        return_dict = {'stds': stds, 'means': means}
        if return_samples:
            return_dict['samples'] = pred_list
        #system('say prediction complete')

        return return_dict
        

class GaussianLRNetworkEnsemble(GaussianNetworkEnsemble):
    def __init__(self,estimator_stats=None,num_epochs=10):

        super(GaussianLRNetworkEnsemble, self).__init__()


        self.estimator_list = [
            GaussianLossEstimator(**x) for x in self.estimator_stats
        ]
