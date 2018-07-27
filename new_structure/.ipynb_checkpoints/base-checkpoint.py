import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = [12, 4]
import scipy

#from networks import EnsembleNetwork, GaussianLearningRateEstimator, GaussianLossEstimator

def safe_ln(x):
	return np.log(x+0.0001)





class BasePredictor(object):
	"""
	https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/05_Ensemble_Learning.ipynb
	"""

	def __init__(self, estimator_stats=None, num_epochs=10):
		pass

	#def train(self, X, y):
	#	pass

	def predict(self, X, return_samples=False):
		pass
	
	def get_prediction_and_std(self,X_test):
		try: 
			pred_dict = self.predict(X_test,True)
			y_hat = pred_dict['means']
			std = pred_dict['stds']
		except:
			
			y_hat = self.predict(X_test)
			#print(pred_dict)
			std = self.predict_var(X_test)
	
		return y_hat,std
	def check_input_dimensions(self,array):
		"""Makes sure arrays are compatible with Tensorflow input
		can't have array.shape = (X,),
		needs to be array.shape = (X,1)"""
		if len(array.shape) == 1:
			return np.expand_dims(array, 1)
		else:
			return array
		
	def network_mutli_dimensional_scatterplot(self,X_test,y_test,X=None,y=None,figsize=(20,50),filename=None):
        
		#y_hat = self.predict(X_test)
		#print(pred_dict)
		#std = self.predict_var(X_test)
		y_hat,std = self.get_prediction_and_std(X_test)
		
		#plt.rcParams["figure.figsize"] = (20,20)
		fig = plt.figure(figsize=figsize)
		#plt.scatter(X[:,5],y)

		num_features = len(X_test.T)
		for i,feature in enumerate(X_test.T):
			#sort the arrays
			s = np.argsort(feature)
			var = y_hat[s]-std[s]
			var2 = y_hat[s] +std[s]
			print(feature.shape)
			print(var.shape)



			plt.subplot(num_features,1,i+1)
			plt.plot(feature[s],y_hat[s],label = 'predictive Mean',)
			plt.fill_between(feature[s].ravel(),y_hat[s].ravel(),var.ravel(),alpha=.3, color='b',label='uncertainty')
			plt.fill_between(feature[s].ravel(),y_hat[s].ravel(),var2.ravel(),alpha=.3, color='b')
			plt.scatter(feature[s],y_test[s],label='data',s=20, edgecolor="black",
				c="darkorange")
			plt.xlabel("data")
			plt.ylabel("target")
			plt.title("Ensemble")
			plt.legend()  
			if filename is not None:
				plt.savefig(filename)

		if filename is not None:
				plt.savefig(filename)
		#plt.show()
		return fig
		
		
	def ensemble_mutli_dimensional_scatterplot(self,X_test,y_test,X=None,y=None,figsize=(20,50),filename=None):
        
		#pred_dict = self.predict(X_test,True)
		#y_hat = pred_dict['means']
		#std = pred_dict['stds']
		y_hat,std = self.get_prediction_and_std(X_test)

		
		#plt.rcParams["figure.figsize"] = (20,20)
		fig = plt.figure(figsize=figsize)
		#plt.scatter(X[:,5],y)

		num_features = len(X_test.T)
		for i,feature in enumerate(X_test.T):
			#sort the arrays
			s = np.argsort(feature)
			var = y_hat[s]-std[s]
			var2 = y_hat[s] +std[s]
			print(feature.shape)
			print(var.shape)



			plt.subplot(num_features,1,i+1)
			plt.plot(feature[s],y_hat[s],label = 'predictive Mean',)
			plt.fill_between(feature[s].ravel(),y_hat[s].ravel(),var.ravel(),alpha=.3, color='b',label='uncertainty')
			plt.fill_between(feature[s].ravel(),y_hat[s].ravel(),var2.ravel(),alpha=.3, color='b')
			plt.scatter(feature[s],y_test[s],label='data',s=20, edgecolor="black",
				c="darkorange")
			plt.xlabel("data")
			plt.ylabel("target")
			plt.title("Ensemble")
			plt.legend()  
			if filename is not None:
				plt.savefig(filename)

		if filename is not None:
				plt.savefig(filename)
		#plt.show()
		return fig

		
	def plot(self, X, y, plot_samples=False, original_func=None,
			 sorted_index=None):
		preds_dict = self.predict(X)
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


	
	#evaluation
	def nlpd(self,X,y):

		y_hat,std = self.get_prediction_and_std(X)

		return -1/2 *np.mean( safe_ln(std) + ((y_hat - y)**2/(std+0.0001)))
	
	def normalised_nlpd(self,X,y):
		pass
		
    
    
	def coverage_probability(self,X, y):

		y_hat,std = self.get_prediction_and_std(X)

		CP = 0
		for pred, s, target in zip(y_hat, std, y):
			#print(len(pred))
			#print(len(s))
			#print(len(target))
			if pred + s > target > pred - s:
				CP += 1
		return CP / len(y)
    
	def error_uncertainty_correlation(self,X,y):
		y_hat,std = self.get_prediction_and_std(X)

		error = np.square(y_hat.flatten() - y.flatten())

		correlation = scipy.stats.pearsonr(error.flatten(),std.flatten())
		

		#np.correlate(error.flatten(),variance.flatten())
		return correlation


	def y_predicts_uncertainty(self,X,y):
		y_hat,std = self.get_prediction_and_std(X)

		correlation = scipy.stats.pearsonr(y_hat.flatten(),y.flatten())
		return correlation


	def y_predicts_error(self,X,y):
		pass

	def error_target_normalcy(self,X,y):
		pass

	def compute_rsme(self,X,y):
		y_hat,std = self.get_prediction_and_std(X)

		return np.sqrt(np.mean((y_hat - y)**2))



	def self_evaluate(self,X,y):

		rsme = self.compute_rsme(X,y)

		cov_prob = self.coverage_probability(X,y)
		err_var_corr = self.error_uncertainty_correlation(X,y)[0]

		nlpd = self.nlpd(X,y)

		return {'rsme':rsme,
				'coverage probability':cov_prob,
			   'correlation between error and variance':err_var_corr,
				'NLPD':nlpd,
			   }

#TODO: SelfEvaluate Network!