import tensorflow as tf
import numpy as np
#tf.reset_default_graph()
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
from os import system
from helpers import lazy_property


class EnsembleNetwork(object):
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
            seed=None):

        #necessary parameters
        self.num_neurons = num_neurons
        self.num_layers = len(num_neurons)
        self.num_features = num_features
        self.learning_rate = learning_rate

        #optional parameters
        self.optimizer = optimizer or tf.train.GradientDescentOptimizer
        self.activations = activations or [tf.nn.tanh] * self.num_layers
        self.initialisation_scheme = initialisation_scheme or tf.random_normal
        self.num_epochs = num_epochs or 1000
        self.seed = seed or None

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

    def train(self, X, y):
        for epoch in range(self.num_epochs):

            self.session.run(self.train_graph,
                             feed_dict={self.X: X,
                                        self.y: y})

    def train_and_evaluate(self, X, y):
        errors = []
        for epoch in range(self.num_epochs):

            self.session.run(self.train_graph,
                             feed_dict={self.X: X,
                                        self.y: y})

            errors.append(
                self.session.run(self.error_graph,
                                 feed_dict={self.X: X,
                                            self.y: y}))

        return errors

    def predict(self, X):
        return self.session.run(self.predict_graph, feed_dict={self.X: X})

    def kill(self):
        self.session.close()


class GaussianLossEstimator(EnsembleNetwork):
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
            seed=None):

        #necessary parameters
        self.num_neurons = num_neurons
        self.num_layers = len(num_neurons)
        self.num_features = num_features
        self.learning_rate = learning_rate

        #optional parameters
        self.optimizer = optimizer or tf.train.GradientDescentOptimizer
        self.activations = activations or [tf.nn.tanh] * self.num_layers
        self.initialisation_scheme = initialisation_scheme or tf.random_normal
        self.num_epochs = num_epochs or 1000
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

    def train(self, X, y):
        for epoch in range(self.num_epochs):
            for batch_X, batch_y in zip(X, y):
                batch_X = np.expand_dims(batch_X, 1)
                batch_y = np.expand_dims(batch_y, 1)
                self.session.run(self.train_graph,
                                 feed_dict={self.X: batch_X,
                                            self.y: batch_y})

    def predict(self, X):
        return self.session.run(self.p_graph, feed_dict={self.X: X})

    def predict_var(self, X):
        return self.session.run(self.std_graph, feed_dict={self.X: X})

    def kill(self):
        self.session.close()


class GaussianLearningRateEstimator(EnsembleNetwork):
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
            seed=None):

        #necessary parameters
        self.num_neurons = num_neurons
        self.num_layers = len(num_neurons)
        self.num_features = num_features

        #optional parameters
        self.optimizer = optimizer or tf.train.GradientDescentOptimizer
        self.activations = activations or [tf.nn.tanh] * self.num_layers
        self.initialisation_scheme = initialisation_scheme or tf.random_normal
        self.num_epochs = num_epochs or 1000
        self.seed = seed or None
        self.learning_rate_init = learning_rate

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

    def train(self, X, y):
        #loss_list =[]
        for epoch in range(self.num_epochs):
            #avg_loss_list = []
            for batch_X, batch_y in zip(X, y):

                batch_X = np.expand_dims(batch_X, 1)
                #if epoch%10 ==0:
                #    avg_loss_list.append(self.get_loss(batch_X))
                batch_y = np.expand_dims(batch_y, 1)
                self.session.run(self.train_graph,
                                 feed_dict={self.X: batch_X,
                                            self.y: batch_y})
            #loss_list.append(np.mean(avg_loss_list))
            #return loss_list

    def predict(self, X):
        #if return_loss:
        #    return {'predictions':self.session.run(self.p_graph,feed_dict={self.X:X}),
        #    'loss':self.session.run(self.error_graph,feed_dict={self.X:X})}

        return self.session.run(self.p_graph, feed_dict={self.X: X})

    def predict_var(self, X):
        return self.session.run(self.std_graph, feed_dict={self.X: X})

    #def get_loss(self,X):
    #    return self.session.run(self.error_graph,feed_dict={self.X:X})

    def kill(self):
        self.session.close()


class DropoutNetwork(EnsembleNetwork):
    def __init__(
            self,
            num_neurons=[10, 10, 10],
            num_features=1,
            keep_prob=0.5,
            learning_rate=0.001,
            activations=None,  #[tf.nn.tanh,tf.nn.relu,tf.sigmoid]
            dropout_layers=None,  #[True,False,True]
            initialisation_scheme=None,  #[tf.random_normal,tf.random_normal,tf.random_normal]
            optimizer=None,  #defaults to GradiendDescentOptimizer,
            num_epochs=None,  #defaults to 1,
            seed=None):

        #necessary parameters
        self.num_neurons = num_neurons
        self.num_layers = len(num_neurons)
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob

        #optional parameters
        self.optimizer = optimizer or tf.train.GradientDescentOptimizer
        self.activations = activations or [tf.nn.tanh] * self.num_layers
        self.initialisation_scheme = initialisation_scheme or tf.random_normal
        self.num_epochs = num_epochs or 1000
        self.seed = seed or None

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

            #z + bias
            if i < self.num_layers:
                bias = self.b_list[i]
                a = tf.add(a, bias)

            #a = sigma(z) if not last layer and regression
            if i < self.num_layers:

                a = self.activations[i](a)
            if i > self.num_layers:
                a = tf.nn.dropout_layers(a, self.keep_prob)
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

    def train(self, X, y):
        for epoch in range(self.num_epochs):

            self.session.run(self.train_graph,
                             feed_dict={self.X: X,
                                        self.y: y})

    def predict(self, X):
        return self.session.run(self.predict_graph, feed_dict={self.X: X})

    def kill(self):
        self.session.close()


# # Data


class GaussianCovProbInvMeanVarEstimator(EnsembleNetwork):
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
            seed=None):

        #necessary parameters
        self.num_neurons = num_neurons
        self.num_layers = len(num_neurons)
        self.num_features = num_features
        self.learning_rate = learning_rate

        #optional parameters
        self.optimizer = optimizer or tf.train.GradientDescentOptimizer
        self.activations = activations or [tf.nn.tanh] * self.num_layers
        self.initialisation_scheme = initialisation_scheme or tf.random_normal
        self.num_epochs = num_epochs or 1000
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
    def cov_prob_graph(self):
        pred = self.p_graph
        std = self.std_graph
        plus = tf.add(pred, std)
        minus = tf.subtract(pred, std)
        greater = tf.greater(plus, self.y)
        less = tf.less(minus, self.y)
        is_it = tf.logical_or(less, greater)
        return tf.cond(is_it, true_fn=lambda: tf.ones([1], tf.float32),
                       false_fn=lambda: tf.zeros([1], tf.float32))

    @lazy_property
    def CovInvVar(self):
        var = self.std_graph
        cov = self.cov_prob_graph
        return tf.multiply(cov, tf.divide(1.0, var))

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
        error = tf.add(
            tf.add(tf.add(first_term, second_term), 1.0), self.CovInvVar)
        #error = 

        return error

    @lazy_property
    def train_graph(self):

        #error is the error from error graph
        error = self.error_graph

        #optimizer is self.optimizer
        optimizer = self.optimizer(learning_rate=self.learning_rate)

        return optimizer.minimize(error)

    def train(self, X, y):
        for epoch in range(self.num_epochs):
            for batch_X, batch_y in zip(X, y):
                batch_X = np.expand_dims(batch_X, 1)
                batch_y = np.expand_dims(batch_y, 1)
                self.session.run(self.train_graph,
                                 feed_dict={self.X: batch_X,
                                            self.y: batch_y})

    def predict(self, X):
        return self.session.run(self.p_graph, feed_dict={self.X: X})

    def predict_var(self, X):
        return self.session.run(self.std_graph, feed_dict={self.X: X})

    def kill(self):
        self.session.close()


# In[4]:
