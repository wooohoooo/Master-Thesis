import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = [12, 4]
import scipy

from estimators import EnsembleNetwork, GaussianLearningRateEstimator, GaussianLossEstimator

def safe_ln(x):
	return np.log(x+0.0001)



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


	def coverage_probability(self,X,y):
		return 42

	def error_uncertainty_correlation(self,X,y):
		return [42]

	def y_predicts_uncertainty(self,X,y):
		return 42
	def compute_rsme(self,X,y):
		return 42

	def nlpd(self,X,y):
		return 42
	
	
	
	
	
	
	
		
	def nlpd(self,X,y):
		pred_dict = self.predict(X,True)
		y_hat = pred_dict['means']
		std = pred_dict['stds']
		
		return -1/2 *np.mean( safe_ln(std) + ((y_hat - y)**2/(std+0.0001)))
	
	def normalised_nlpd(self,X,y):
		pass
		
    
    
	def coverage_probability(self,X, y):

		pred_dict = self.predict(X,True)
		y_hat = pred_dict['means']
		std = pred_dict['stds']		#print(y_hat.shape,std.shape,y.shape)

		CP = 0
		for pred, s, target in zip(y_hat, std, y):
			#print(len(pred))
			#print(len(s))
			#print(len(target))
			if pred + s > target > pred - s:
				CP += 1
		return CP / len(y)
    
	def error_uncertainty_correlation(self,X,y):
		pred_dict = self.predict(X,True)
		y_hat = pred_dict['means']
		std = pred_dict['stds']		#print(y_hat.shape,std.shape,y.shape)

		error = (y_hat - y)**2
		correlation = scipy.stats.pearsonr(error.flatten(),std.flatten())

		#np.correlate(error.flatten(),variance.flatten())
		return correlation
	



	def y_predicts_uncertainty(self,X,y):
		pred_dict = self.predict(X,True)
		y_hat = pred_dict['means']
		std = pred_dict['stds']		#print(y_hat.shape,std.shape,y.shape)

		correlation = scipy.stats.pearsonr(y_hat.flatten(),y.flatten())
		return correlation


	def y_predicts_error(self,X,y):
		pass

	def error_target_normalcy(self,X,y):
		pass

	def compute_rsme(self,X,y):
		pred_dict = self.predict(X,True)
		y_hat = pred_dict['means']
		std = pred_dict['stds']		#print(y_hat.shape,std.shape,y.shape)

		return np.sqrt(np.mean((y_hat - y)**2))



	def self_evaluate(self,X,y):

		rsme = self.compute_rsme(X,y)

		cov_prob = self.coverage_probability(X,y)
		#print('coverage Probability is: {}'.format(cov_prob))
		err_var_corr = self.error_uncertainty_correlation(X,y)[0]
		#print('correlation of error and uncertainty is: {}'.format(err_var_corr)) #0 is the coefficient
		#y_uncertainty_pred = self.y_predicts_uncertainty(X,y)[0]
		#print('correlation of target value and uncertainty is: {}'.format(y_uncertainty_pred)) #0 is the coefficient
		#y_predicts_error = self.y_predicts_error(X,y)[0]
		#print('correlation of target value and error is: {}'.format(y_uncertainty_pred)) #0 is the coefficient
		#target_error_normalcy = self.error_target_normalcy(X,y)[0]
		#print('error-target normalcy is {}'.format(target_error_normalicy))
		nlpd = self.nlpd(X,y)

		return {'rsme':rsme,
				'coverage probability':cov_prob,
			   'correlation between error and variance':err_var_corr,
				'NLPD':nlpd,
			   #'predictive power of y on the uncertainty':y_uncertainty_pred,
			   #'predictive power of y on the error': y_predicts_error,
			   #'error normalcy':target_error_normalcy
			   }


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
