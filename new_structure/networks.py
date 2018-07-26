import tensorflow as tf
import numpy as np
#tf.reset_default_graph()
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
from os import system
from helpers import lazy_property, unison_shuffled_copies
#from datasets import unison_shuffled_copies

from global_vars import ADVERSARIAL

import base
import importlib
importlib.reload(base) #this is for notebooks - otherwise I have to restart the kernel every time I change anything

class EnsembleNetwork(base.BasePredictor,object):
	def __init__(
			self,
			num_neurons=[10, 10, 10],
			num_features=1,
			learning_rate=0.001,
			activations=None,  #[tf.nn.tanh,tf.nn.relu,tf.sigmoid]
			dropout_layers=None,  #[True,False,True]
			initialisation_scheme=None,  #[tf.random_normal,tf.random_normal,tf.random_normal]
			optimizer=None,  #defaults to GradiendDescentOptimizer,
			num_epochs=None,  #defaults to 1,
			seed=None,
			adversarial=None):

		#necessary parameters
		self.num_neurons = num_neurons
		self.num_layers = len(num_neurons)
		self.num_features = num_features
		self.learning_rate = learning_rate
		self.adversarial = adversarial or ADVERSARIAL

		#optional parameters
		self.optimizer = optimizer or tf.train.GradientDescentOptimizer
		self.activations = activations or [tf.nn.tanh] * self.num_layers
		self.initialisation_scheme = initialisation_scheme or tf.random_normal
		self.num_epochs = num_epochs or 1000
		self.seed = seed or None

		self.initialise_graph
		self.initialise_session

	@lazy_property
	def initialise_graph(self):
		#initialise graph
		self.g = tf.Graph()
		#build graph with self.graph as default so nodes get appended
		with self.g.as_default():
			self.init_network
			self.predict_graph
			self.error_graph
			self.train_graph
			self.init = tf.global_variables_initializer()

	@lazy_property
	def initialise_session(self):
		#initialise session
		self.session = tf.Session(graph=self.g)
		#initialise global variables
		self.session.run(self.init)

	@lazy_property
	def init_network(self):
		if self.seed:
			tf.set_random_seed(self.seed)
		#inputs
		self.X = tf.placeholder(tf.float32, (None, self.num_features))
		self.y = tf.placeholder(tf.float32, (None, 1))  #regression = 1

		#lists for storage
		self.w_list = []
		self.b_list = []

		#add input x first weights
		self.w_list.append(
			tf.Variable(
				self.initialisation_scheme(
					[self.num_features, self.num_neurons[0]]),
				name='w_0'))  #first Matrix

		#for each layer over 0 add a n x m matrix and a bias term
		for i, num_neuron in enumerate(self.num_neurons[1:]):
			n_inputs = self.num_neurons[i]  #for first hidden layer 3
			n_outputs = self.num_neurons[i + 1]  #for first hidden layer 5

			self.w_list.append(
				tf.Variable(
					self.initialisation_scheme([n_inputs, n_outputs]),
					name='w_' + str(i)))
			self.b_list.append(
				tf.Variable(tf.ones(shape=[n_inputs]), name='b_' + str(i)))

		#add last layer m  x 1 for output
		self.w_list.append(
			tf.Variable(
				self.initialisation_scheme([self.num_neurons[-1], 1]),
				name='w_-1'))  #this is a regression
		self.b_list.append(
			tf.Variable(
				tf.ones(shape=[self.num_neurons[-1]]), name='b_' + str(
					len(self.num_neurons) + 1)))

	@lazy_property
	def predict_graph(self):
		#set layer_input to input
		layer_input = self.X

		#for each layer do
		for i, w in enumerate(self.w_list):

			#z = input x Weights
			a = tf.matmul(layer_input, w, name='matmul_' + str(i))

			#z + bias
			if i < self.num_layers:
				bias = self.b_list[i]
				a = tf.add(a, bias)

			#a = sigma(z) if not last layer and regression
			if i < self.num_layers:

				a = self.activations[i](a)
			#set layer input to a for next cycle
			layer_input = a

		return a

	@lazy_property
	def error_graph(self):

		#y_hat is a // output of prediction graph
		y_hat = self.predict_graph

		#error is mean squared error of placehilder y and prediction
		error = tf.losses.mean_squared_error(self.y, y_hat)  #tf.square(
		#self.y - y_hat)  #

		return error

	@lazy_property
	def train_graph(self):

		#error is the error from error graph
		error = self.error_graph

		#optimizer is self.optimizer
		optimizer = self.optimizer(learning_rate=self.learning_rate)

		return optimizer.minimize(error)

	def train_offline(self, X, y):
		for epoch in range(self.num_epochs):

			self.session.run(self.train_graph,
							 feed_dict={self.X: X,
										self.y: y})

	def train(self, X, y, shuffle=True, online=False):
		#print('X is {}'.format(X[:10]))

		X = self.check_input_dimensions(X)
		y = self.check_input_dimensions(y)
		if online:
			for epoch in range(self.num_epochs):
				self.train_one_epoch(X, y, shuffle)
		else:
			self.train_offline(X, y)

	def train_one_epoch(self, epoch_X, epoch_y, shuffle=True):
		if shuffle == True:
			epoch_X, epoch_y, _ = unison_shuffled_copies(
				np.squeeze(epoch_X), np.squeeze(epoch_y), True)
		#print(epoch_X[:10])

		for batch_X, batch_y in zip(epoch_X, epoch_y):
			self.train_one(batch_X, batch_y)
			if self.adversarial:
				self.train_one(batch_X+0.05, batch_y)
				self.train_one(batch_X-0.05, batch_y)

	def train_one(self, batch_X, batch_y):
		batch_X = self.check_input_dimensions(batch_X)
		batch_y = self.check_input_dimensions(batch_y)
		#batch_X = np.expand_dims(batch_X, 1)
		#batch_y = np.expand_dims(batch_y, 1)
		self.session.run(self.train_graph,
						 feed_dict={self.X: batch_X,
									self.y: batch_y})

	def train_and_evaluate(self, X, y, shuffle=False):
		errors = []
		for epoch in range(self.num_epochs):

			#self.session.run(self.train_graph,
			#                 feed_dict={self.X: X,
			#                            self.y: y})
			self.train_one_epoch(X, y, shuffle)



			errors += list(np.sqrt((y - self.predict(X))**2))

			#self.session.run(self.predict_graph,
			#                          feed_dict={self.X: X,
			#                                     self.y: y}))**2))

		return errors

	def predict(self, X):
		return self.session.run(self.predict_graph, feed_dict={self.X: X})

	def kill(self):
		self.session.close()

		
class GaussianLossNetwork(EnsembleNetwork):
	def __init__(
			self,
			num_neurons=[10, 10, 10],
			num_features=1,
			learning_rate=0.001,
			activations=None,  #[tf.nn.tanh,tf.nn.relu,tf.sigmoid]
			dropout_layers=None,  #[True,False,True]
			initialisation_scheme=None,  #[tf.random_normal,tf.random_normal,tf.random_normal]
			optimizer=None,  #defaults to GradiendDescentOptimizer,
			num_epochs=None,  #defaults to 1,
			seed=None,
			adversarial=None):

		#necessary parameters
		self.num_neurons = num_neurons
		self.num_layers = len(num_neurons)
		self.num_features = num_features
		self.learning_rate = learning_rate
		self.adversarial = adversarial or ADVERSARIAL

		#optional parameters
		self.optimizer = optimizer or tf.train.GradientDescentOptimizer
		self.activations = activations or [tf.nn.tanh] * self.num_layers
		self.initialisation_scheme = initialisation_scheme or tf.random_normal
		self.num_epochs = num_epochs or 10
		self.seed = seed or None

		#initialise graph
		self.g = tf.Graph()
		#build graph with self.graph as default so nodes get appended
		with self.g.as_default():
			self.init_network
			self.predict_graph
			self.p_graph
			self.std_graph
			self.error_graph
			self.train_graph
			self.init = tf.global_variables_initializer()

		#initialise session
		self.session = tf.Session(graph=self.g)
		#initialise global variables
		self.session.run(self.init)

	@lazy_property
	def init_network(self):
		if self.seed:
			tf.set_random_seed(self.seed)
		#inputs
		self.X = tf.placeholder(tf.float32, (None, self.num_features))
		self.y = tf.placeholder(tf.float32, (None, 1))  #regression = 1

		#lists for storage
		self.w_list = []
		self.b_list = []

		#add input x first weights
		self.w_list.append(
			tf.Variable(
				self.initialisation_scheme(
					[self.num_features, self.num_neurons[0]]),
				name='w_0'))  #first Matrix

		#for each layer over 0 add a n x m matrix and a bias term
		for i, num_neuron in enumerate(self.num_neurons[1:]):
			n_inputs = self.num_neurons[i]  #for first hidden layer 3
			n_outputs = self.num_neurons[i + 1]  #for first hidden layer 5

			self.w_list.append(
				tf.Variable(
					self.initialisation_scheme([n_inputs, n_outputs]),
					name='w_' + str(i)))
			self.b_list.append(
				tf.Variable(tf.ones(shape=[n_inputs]), name='b_' + str(i)))

		#add last layer m  x 1 for output
		self.p_w = tf.Variable(
			self.initialisation_scheme([self.num_neurons[-1], 1]),
			name='w_-1')  #this is a regression
		self.std_w = tf.Variable(
			self.initialisation_scheme([self.num_neurons[-1], 1]),
			name='w_std')  #this is a regression

		self.b_list.append(
			tf.Variable(
				tf.ones(shape=[self.num_neurons[-1]]), name='b_' + str(
					len(self.num_neurons) + 1)))

	@lazy_property
	def predict_graph(self):
		#set layer_input to input
		layer_input = self.X

		#for each layer do
		for i, w in enumerate(self.w_list):

			#z = input x Weights
			a = tf.matmul(layer_input, w, name='matmul_' + str(i))

			#z + bias
			#if i < self.num_layers:
			bias = self.b_list[i]
			a = tf.add(a, bias)

			#a = sigma(z) if not last layer and regression
			#if i < self.num_layers:    

			a = self.activations[i](a)

			#set layer input to a for next cycle
			layer_input = a

		return a

	@lazy_property
	def p_graph(self):
		l_in = self.predict_graph
		return tf.matmul(l_in, self.p_w)

	@lazy_property
	def std_graph(self):
		l_in = self.predict_graph
		return tf.nn.softplus(tf.matmul(l_in, self.std_w))

	@lazy_property
	def error_graph(self):

		#y_hat is a // output of prediction graph
		y_hat = self.p_graph
		std_hat = tf.maximum(self.std_graph, 0.005)  #tf.square(self.std_graph)
		#s_hat = self.predict_graph[1]

		#error is mean squared error of placehilder y and prediction
		#error = tf.losses.mean_squared_error(self.y,y_hat)
		#first_term = tf.div(tf.square(std_hat),2.0)
		#second_term = tf.div(tf.square(tf.subtract(self.y,y_hat)), tf.multiply(2.0,std_hat) )
		first_term = tf.log(tf.div(std_hat, 2.0))
		second_term = tf.div(
			tf.square(tf.subtract(self.y, y_hat)), tf.multiply(2.0, std_hat))
		#error = -tf.log(tf.add(first_term,second_term))
		error = tf.add(tf.add(first_term, second_term), 1.0)
		#error = 

		return error

	@lazy_property
	def train_graph(self):

		#error is the error from error graph
		error = self.error_graph

		#optimizer is self.optimizer
		optimizer = self.optimizer(learning_rate=self.learning_rate)

		return optimizer.minimize(error)

	#def train(self, X, y):
	#    for epoch in range(self.num_epochs):
	#        for batch_X, batch_y in zip(X, y):
	#            batch_X = np.expand_dims(batch_X, 1)
	#            batch_y = np.expand_dims(batch_y, 1)
	#            self.session.run(self.train_graph,
	#                             feed_dict={self.X: batch_X,
	#                                        self.y: batch_y})

	#def train_and_evaluate_old(self, X, y):
	#    errors = []
	#    for epoch in range(self.num_epochs):#

	#          for batch_X, batch_y in zip(X, y):
	#              batch_X = np.expand_dims(batch_X, 1)
	#              batch_y = np.expand_dims(batch_y, 1)
	#              self.session.run(self.train_graph,
	#                               feed_dict={self.X: batch_X,
	#                                          self.y: batch_y})

	#             errors += self.session.run(self.error_graph, feed_dict={
	#                 self.X: batch_X,
	#                 self.y: batch_y
	#             })
	#     return errors

	#def train_and_evaluate_old(self, X, y):
	#    errors = []
	#    for epoch in range(self.num_epochs):#

	#       for batch_X, batch_y in zip(X, y):#
	#           batch_X = np.expand_dims(batch_X, 1)
	#           batch_y = np.expand_dims(batch_y, 1)
	#           self.session.run(self.train_graph,
	#                            feed_dict={self.X: batch_X,
	#                                       self.y: batch_y})

	#           y_hat = self.session.run(
	#               self.p_graph, feed_dict={self.X: batch_X,
	#                                        self.y: batch_y})
	#           error = np.sqrt((y_hat - batch_y)**2)  #RMSE
	#           errors.append(error)
	#   return errors

	def fit(self, X, y, online=True):

		X = self.check_input_dimensions(X)
		y = self.check_input_dimensions(y)
		super(GaussianLossNetwork, self).train(X, y, online=online)

	def predict(self, X):
		return self.session.run(self.p_graph, feed_dict={self.X: X})

	def predict_var(self, X):
		return self.session.run(self.std_graph, feed_dict={self.X: X})

	#def kill(self):
	#    self.session.close()



class GaussianLearningRateNetwork(GaussianLossNetwork):
	def __init__(
			self,
			num_neurons=[10, 10, 10],
			num_features=1,
			learning_rate=0.001,
			activations=None,  #[tf.nn.tanh,tf.nn.relu,tf.sigmoid]
			dropout_layers=None,  #[True,False,True]
			initialisation_scheme=None,  #[tf.random_normal,tf.random_normal,tf.random_normal]
			optimizer=None,  #defaults to GradiendDescentOptimizer,
			num_epochs=None,  #defaults to 1,
			seed=None,
			adversarial=None):

		#necessary parameters
		self.num_neurons = num_neurons
		self.num_layers = len(num_neurons)
		self.num_features = num_features

		#optional parameters
		self.optimizer = optimizer or tf.train.GradientDescentOptimizer
		self.activations = activations or [tf.nn.tanh] * self.num_layers
		self.initialisation_scheme = initialisation_scheme or tf.random_normal
		self.num_epochs = num_epochs or 10
		self.seed = seed or None
		self.learning_rate_init = learning_rate
		self.adversarial = adversarial or ADVERSARIAL

		#initialise graph
		self.g = tf.Graph()
		#build graph with self.graph as default so nodes get appended
		with self.g.as_default():

			self.init_network
			self.predict_graph
			self.p_graph
			self.std_graph
			self.error_graph
			self.train_graph
			self.init = tf.global_variables_initializer()

		#initialise session
		self.session = tf.Session(graph=self.g)
		#initialise global variables
		self.session.run(self.init)

	@lazy_property
	def init_network(self):
		if self.seed:
			tf.set_random_seed(self.seed)
		#inputs
		self.X = tf.placeholder(tf.float32, (None, self.num_features))
		self.y = tf.placeholder(tf.float32, (None, 1))  #regression = 1

		#learning rate Variable
		self.learning_rate = tf.Variable(self.learning_rate_init)

		#lists for storage
		self.w_list = []
		self.b_list = []

		#add input x first weights
		self.w_list.append(
			tf.Variable(
				self.initialisation_scheme(
					[self.num_features, self.num_neurons[0]]),
				name='w_0'))  #first Matrix

		#for each layer over 0 add a n x m matrix and a bias term
		for i, num_neuron in enumerate(self.num_neurons[1:]):
			n_inputs = self.num_neurons[i]  #for first hidden layer 3
			n_outputs = self.num_neurons[i + 1]  #for first hidden layer 5

			self.w_list.append(
				tf.Variable(
					self.initialisation_scheme([n_inputs, n_outputs]),
					name='w_' + str(i)))
			self.b_list.append(
				tf.Variable(tf.ones(shape=[n_inputs]), name='b_' + str(i)))

		#add last layer m  x 1 for output
		self.p_w = tf.Variable(
			self.initialisation_scheme([self.num_neurons[-1], 1]),
			name='w_-1')  #this is a regression
		self.std_w = tf.Variable(
			self.initialisation_scheme([self.num_neurons[-1], 1]),
			name='w_std')  #this is a regression

		self.b_list.append(
			tf.Variable(
				tf.ones(shape=[self.num_neurons[-1]]), name='b_' + str(
					len(self.num_neurons) + 1)))

	@lazy_property
	def predict_graph(self):
		#set layer_input to input
		layer_input = self.X

		#for each layer do
		for i, w in enumerate(self.w_list):

			#z = input x Weights
			a = tf.matmul(layer_input, w, name='matmul_' + str(i))

			#z + bias
			#if i < self.num_layers:
			bias = self.b_list[i]
			a = tf.add(a, bias)

			#a = sigma(z) if not last layer and regression
			#if i < self.num_layers:    

			a = self.activations[i](a)

			#set layer input to a for next cycle
			layer_input = a

		return a

	@lazy_property
	def p_graph(self):
		l_in = self.predict_graph
		return tf.matmul(l_in, self.p_w)

	@lazy_property
	def std_graph(self):
		l_in = self.predict_graph
		return tf.nn.softplus(tf.matmul(l_in, self.std_w))

	@lazy_property
	def error_graph(self):

		#y_hat is a // output of prediction graph
		y_hat = self.p_graph
		std_hat = tf.maximum(self.std_graph, 0.001)  #tf.square(self.std_graph)
		#s_hat = self.predict_graph[1]

		#error is mean squared error of placehilder y and prediction
		#error = tf.losses.mean_squared_error(self.y,y_hat)
		#first_term = tf.div(tf.square(std_hat),2.0)
		#second_term = tf.div(tf.square(tf.subtract(self.y,y_hat)), tf.multiply(2.0,std_hat) )
		first_term = tf.log(tf.div(std_hat, 2.0))
		second_term = tf.div(
			tf.square(tf.subtract(self.y, y_hat)), tf.multiply(2.0, std_hat))
		#error = -tf.log(tf.add(first_term,second_term))
		error = tf.add(tf.add(first_term, second_term), 1.0)
		#error = 

		return error

	@lazy_property
	def learning_rate_graph(self):
		learning_rate = tf.squeeze(
			tf.multiply(self.learning_rate,
						tf.maximum(tf.sqrt(self.std_graph), 0.05)))
		return learning_rate

	@lazy_property
	def train_graph(self):

		#error is the error from error graph
		error = self.error_graph

		#optimizer is self.optimizer
		optimizer = self.optimizer(learning_rate=self.learning_rate_graph)

		return optimizer.minimize(error)

	#def train(self, X, y):
	#    #loss_list =[]
	#    for epoch in range(self.num_epochs):
	#        #avg_loss_list = []
	#        for batch_X, batch_y in zip(X, y):#

	#            batch_X = np.expand_dims(batch_X, 1)
	#if epoch%10 ==0:
	#    avg_loss_list.append(self.get_loss(batch_X))
	#            batch_y = np.expand_dims(batch_y, 1)
	#            self.session.run(self.train_graph,
	#                             feed_dict={self.X: batch_X,
	#                                        self.y: batch_y})
	#loss_list.append(np.mean(avg_loss_list))
	#return loss_list
	#def train_and_evaluate_old(self, X, y):
	#    errors = []
	#    for epoch in range(self.num_epochs):#

	#           for batch_X, batch_y in zip(X, y):
	#              batch_X = np.expand_dims(batch_X, 1)
	#             batch_y = np.expand_dims(batch_y, 1)
	#            self.session.run(self.train_graph,
	#                            feed_dict={self.X: batch_X,
	#                                            self.y: batch_y})

	#               y_hat = self.session.run(self.predict_graph, feed_dict={
	#                  self.X: batch_X,
	#                 self.y: batch_y
	#            })
	#            error = np.sqrt((y_hat - batch_y)**2)  #RMSE
	#           errors.append(error)
	#  return errors

	#def predict(self, X):
	#if return_loss:
	#    return {'predictions':self.session.run(self.p_graph,feed_dict={self.X:X}),
	#    'loss':self.session.run(self.error_graph,feed_dict={self.X:X})}

	#    return self.session.run(self.p_graph, feed_dict={self.X: X})

	#def predict_var(self, X):
	#    return self.session.run(self.std_graph, feed_dict={self.X: X})

	#def get_loss(self,X):
	#    return self.session.run(self.error_graph,feed_dict={self.X:X})

	#def kill(self):
	#    self.session.close()


	
	
	  
        
class DropoutNetwork(EnsembleNetwork):
	def __init__(
			self,
			num_neurons=[10, 10, 10],
			num_features=1,
			keep_prob=0.8,
			learning_rate=0.001,
			activations=None,  #[tf.nn.tanh,tf.nn.relu,tf.sigmoid]
			dropout_layers=None,#[2],#]None,  #[1,3,5]#[True,False,True]
			initialisation_scheme=None,  #[tf.random_normal,tf.random_normal,tf.random_normal]
			optimizer=None,  #defaults to GradiendDescentOptimizer,
			num_epochs=None,  #defaults to 1,
			seed=None,
			num_samples=None,
			adversarial = None):

		#necessary parameters
		self.num_neurons = num_neurons
		self.num_layers = len(num_neurons)
		self.num_features = num_features
		self.learning_rate = learning_rate
		self.keep_prob = keep_prob
		self.sample_list = None
		self.dropout_layers = dropout_layers or [self.num_layers]#[num_layers]

		#optional parameters
		self.optimizer = optimizer or tf.train.GradientDescentOptimizer
		self.activations = activations or [tf.sigmoid] * self.num_layers #[tf.nn.tanh]
		self.initialisation_scheme = initialisation_scheme or tf.random_normal
		self.num_epochs = num_epochs or 1000
		self.seed = seed or None
		self.num_samples = num_samples or 10
		self.adversarial = adversarial or ADVERSARIAL

		#initialise graph
		self.g = tf.Graph()
		#build graph with self.graph as default so nodes get appended
		with self.g.as_default():
			self.init_network
			self.predict_graph
			self.error_graph
			self.train_graph
			self.init = tf.global_variables_initializer()

		#initialise session
		self.session = tf.Session(graph=self.g)
		#initialise global variables
		self.session.run(self.init)

	@lazy_property
	def init_network(self):
		if self.seed:
			tf.set_random_seed(self.seed)
		#inputs
		self.X = tf.placeholder(tf.float32, (None, self.num_features))
		self.y = tf.placeholder(tf.float32, (None, 1))  #regression = 1

		#lists for storage
		self.w_list = []
		self.b_list = []

		#add input x first weights
		self.w_list.append(
			tf.Variable(
				self.initialisation_scheme(
					[self.num_features, self.num_neurons[0]]),
				name='w_0'))  #first Matrix

		#for each layer over 0 add a n x m matrix and a bias term
		for i, num_neuron in enumerate(self.num_neurons[1:]):
			n_inputs = self.num_neurons[i]  #for first hidden layer 3
			n_outputs = self.num_neurons[i + 1]  #for first hidden layer 5

			self.w_list.append(
				tf.Variable(
					self.initialisation_scheme([n_inputs, n_outputs]),
					name='w_' + str(i)))
			self.b_list.append(
				tf.Variable(tf.ones(shape=[n_inputs]), name='b_' + str(i)))

		#add last layer m  x 1 for output
		self.w_list.append(
			tf.Variable(
				self.initialisation_scheme([self.num_neurons[-1], 1]),
				name='w_-1'))  #this is a regression
		self.b_list.append(
			tf.Variable(
				tf.ones(shape=[self.num_neurons[-1]]), name='b_' + str(
					len(self.num_neurons) + 1)))

	@lazy_property
	def predict_graph(self):
		#set layer_input to input
		layer_input = self.X

		#for each layer do
		for i, w in enumerate(self.w_list):

			#z = input x Weights
			a = tf.matmul(layer_input, w, name='matmul_' + str(i))

			#add dropout
			if i in self.dropout_layers:#== self.num_layers:
				a = tf.nn.dropout(a, self.keep_prob)

			#z + bias
			if i < self.num_layers:
				bias = self.b_list[i]
				a = tf.add(a, bias)

			#a = sigma(z) if not last layer and regression
			if i < self.num_layers:

				a = self.activations[i](a)

			#set layer input to a for next cycle
			layer_input = a

		return a

	@lazy_property
	def error_graph(self):

		#y_hat is a // output of prediction graph
		y_hat = self.predict_graph

		#error is mean squared error of placehilder y and prediction
		error = tf.losses.mean_squared_error(self.y, y_hat)

		return error

	@lazy_property
	def train_graph(self):

		#error is the error from error graph
		error = self.error_graph

		#optimizer is self.optimizer
		optimizer = self.optimizer(learning_rate=self.learning_rate)

		return optimizer.minimize(error)

	def fit(self, X, y):
		X = self.check_input_dimensions(X)
		y = self.check_input_dimensions(y)
		for epoch in range(self.num_epochs):

			self.session.run(self.train_graph,
							 feed_dict={self.X: X,
										self.y: y})



	def predict_one(self, X): #predict_one
		return self.session.run(self.predict_graph, feed_dict={self.X: X})

	def generate_samples(self,X):
		#if self.sample_list = None:
		sample_list = []
		for i in range(self.num_samples):
			sample_list.append(self.predict_one(X)) #predict_one
		#    self.sample_list = sample_list
		#return self.sample_list
		return sample_list

	def predict(self,X):
		mean = np.mean(self.generate_samples(X),axis=0)
		#print(var)
		return mean


	def predict_var(self,X):
		var = np.std(self.generate_samples(X),axis=0)
		#print(var)
		return var




	def kill(self):
		self.session.close()

	
	