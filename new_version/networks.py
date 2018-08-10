import base
import importlib
importlib.reload(base)
from helpers import lazy_property
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(0)


class CopyNetwork(base.EnsembleNetwork):
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

        super(CopyNetwork, self).__init__(
            num_neurons, num_features, learning_rate, activations,
            dropout_layers, initialisation_scheme, optimizer, num_epochs, seed,
            adversarial)

        self.save_names = []

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
            self.saver = tf.train.Saver(
                max_to_keep=None
            )  #make sure none of the models are discarded without explicitly wanting to

    def save(self, name='testname'):
        #with self.session as sess:

        #self.session.run(self.saver.save(self.session, name))
        self.saver.save(self.session, name)

        self.save_names.append(name)

    def load(self, name='testname'):

        self.saver.restore(self.session, name)

        #new_saver = tf.train.import_meta_graph(name)
        #new_saver.restore(self.session, tf.train.latest_checkpoint('./'))


class DropoutNetwork(base.EnsembleNetwork):
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
            adversarial=None,
            l2=None,
            l=None,
            num_preds=None,
            keep_prob=None):

        self.num_preds = num_preds or 50
        self.keep_prob = keep_prob or 0.85

        super(DropoutNetwork, self).__init__(
            num_neurons=num_neurons, num_features=num_features,
            learning_rate=learning_rate, activations=activations,
            dropout_layers=dropout_layers,
            initialisation_scheme=initialisation_scheme, optimizer=optimizer,
            num_epochs=num_epochs, seed=seed, adversarial=adversarial, l2=l2,
            l=l)

    @lazy_property
    def predict_graph(self):
        #set layer_input to input
        layer_input = self.X

        #for each layer do
        for i, w in enumerate(self.w_list):

            #z = input x Weights
            a = tf.matmul(layer_input, w, name='matmul_' + str(i))

            if i == self.num_layers:  #This is new - Dropout!
                a = tf.nn.dropout(a, self.keep_prob)  #0.9 = keep_prob

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

    def predict(self, X):
        X = self.check_input_dimensions(X)

        pred_list = [
            self.session.run(self.predict_graph,
                             feed_dict={self.X: X}).squeeze()
            for i in range(self.num_preds)
        ]

        pred_mean = np.mean(pred_list, axis=0)
        #pred_std = np.std(pred_list,axis=0)
        return pred_mean  #, pred_std

    def get_mean_and_std(self, X):
        #compute tau http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html
        num_datapoints = len(X)
        length_scale = 1
        tau = (length_scale**2 * self.keep_prob) / (2 * num_datapoints *
                                                    self.l)

        X = self.check_input_dimensions(X)

        pred_list = [
            self.session.run(self.predict_graph,
                             feed_dict={self.X: X}).squeeze()
            for i in range(15)
        ]

        pred_mean = np.mean(pred_list, axis=0)
        pred_std = np.var(pred_list, axis=0) + tau  #**-1
        pred_std[pred_std == 0] = 0.01
        return pred_mean, pred_std


class NlpdNetwork(base.EnsembleNetwork):
    #TODO: Move parameter initialisation into base so that changes there affect this Network, too
    def __init__(
            self,
            num_neurons=[10, 5, 5, 5, 5],
            num_features=1,
            learning_rate=0.001,
            activations=None,  #[tf.nn.tanh,tf.nn.relu,tf.sigmoid]
            dropout_layers=None,  #[True,False,True]
            initialisation_scheme=None,  #[tf.random_normal,tf.random_normal,tf.random_normal]
            optimizer=None,  #defaults to GradiendDescentOptimizer,
            num_epochs=None,  #defaults to 1,
            seed=None,
            adversarial=None,
            initialisation_params=None,
            l2=None):

        #necessary parameters
        self.num_neurons = num_neurons
        self.num_layers = len(num_neurons)
        self.num_features = num_features
        self.learning_rate = learning_rate or 0.001
        self.adversarial = adversarial or False
        self.initialisation_params = initialisation_params or {}
        self.l2 = l2 or None

        #optional parameters
        self.optimizer = optimizer or tf.train.AdamOptimizer  #tf.train.GradientDescentOptimizer
        self.activations = activations or [tf.nn.relu
                                           ] * self.num_layers  #tanh,relu, 
        self.initialisation_scheme = initialisation_scheme or tf.contrib.layers.xavier_initializer  #tf.keras.initializers.he_normal  #
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
        self.min_std = tf.Variable(0.05)  #THIS IS NEW

        if self.seed:
            tf.set_random_seed(self.seed)
        #inputs
        self.X = tf.placeholder(tf.float32, (None, self.num_features))
        self.y = tf.placeholder(tf.float32, (None, 1))  #regression = 1

        #lists for storage
        self.w_list = []
        self.b_list = []

        initialiser = self.initialisation_scheme(seed=self.seed,
                                                 **self.initialisation_params)
        #add input x first weights
        self.w_list.append(
            tf.Variable(
                initialiser([self.num_features, self.num_neurons[0]]),
                name='w_0'))  #first Matrix

        #for each layer over 0 add a n x m matrix and a bias term
        for i, num_neuron in enumerate(self.num_neurons[1:]):
            n_inputs = self.num_neurons[i]  #for first hidden layer 3
            n_outputs = self.num_neurons[i + 1]  #for first hidden layer 5

            self.w_list.append(
                tf.Variable(
                    initialiser([n_inputs, n_outputs]), name='w_' + str(i)))
            self.b_list.append(
                tf.Variable(tf.ones(shape=[n_inputs]), name='b_' + str(i)))
        #THis is new: Build two separate outputs for mean and std
        self.p_w = tf.Variable(
            initialiser([self.num_neurons[-1], 1]),
            name='w_-1')  #this is a regression
        self.std_w = tf.Variable(
            initialiser([self.num_neurons[-1], 1]),
            name='w_std')  #this is a regression

        self.b_list.append(
            tf.Variable(
                tf.ones(shape=[self.num_neurons[-1]]), name='b_' + str(
                    len(self.num_neurons) + 1)))

    @lazy_property
    def predict_graph_old(self):
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
        std_hat = tf.maximum(self.std_graph,
                             self.min_std)  #tf.square(self.std_graph)

        #first_term = tf.log(tf.div(std_hat, 2.0))
        first_term = tf.log(std_hat)
        second_term = tf.div(tf.square(tf.subtract(self.y, y_hat)), std_hat)
        error = tf.add(tf.add(first_term, second_term), 1.0)

        if self.l2:
            # Loss function with L2 Regularization with beta=0.01
            regularizers = tf.reduce_sum(
                [tf.nn.l2_loss(weights) for weights in self.w_list])
            error = tf.reduce_mean(error + 0.01 * regularizers)

        return error

    def predict(self, X):
        X = self.check_input_dimensions(X)

        return self.session.run(self.p_graph, feed_dict={
            self.X: X
        })  # now calls p_graph instead of predict_graph

    def predict_var(self, X):
        X = self.check_input_dimensions(X)

        return self.session.run(self.std_graph, feed_dict={self.X: X})

    def get_mean_and_std(self, X):
        return self.predict(X).flatten(), self.predict_var(X).flatten()


class LrNetwork(NlpdNetwork):
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
            adversarial=None,
            initialisation_params=None,
            l2=None):

        #necessary parameters
        self.num_neurons = num_neurons
        self.num_layers = len(num_neurons)
        self.num_features = num_features
        self.l2 = l2 or False

        #optional parameters
        #self.optimizer = optimizer or tf.train.GradientDescentOptimizer
        #self.activations = activations or [tf.nn.tanh] * self.num_layers
        #self.initialisation_scheme = initialisation_scheme or tf.random_normal
        self.optimizer = optimizer or tf.train.AdamOptimizer  #tf.train.GradientDescentOptimizer
        self.activations = activations or [tf.nn.relu
                                           ] * self.num_layers  #tanh,relu, 
        self.initialisation_scheme = initialisation_scheme or tf.contrib.layers.xavier_initializer  #tf.keras.initializers.he_normal  #
        self.num_epochs = num_epochs or 10
        self.seed = seed or None
        self.learning_rate_init = learning_rate or 0.001
        self.adversarial = adversarial or False
        self.initialisation_params = initialisation_params or {}

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
            self.learning_rate_graph
            self.init = tf.global_variables_initializer()

        #initialise session
        self.session = tf.Session(graph=self.g)
        #initialise global variables
        self.session.run(self.init)

    @lazy_property
    def init_lr(self):
        self.learning_rate = tf.Variable(self.learning_rate_init)

    @lazy_property
    def learning_rate_graph(self):
        learning_rate = tf.squeeze(
            tf.multiply(self.learning_rate_init,
                        tf.maximum(tf.sqrt(self.std_graph), 0.05)))
        return learning_rate

    @lazy_property
    def train_graph(self):

        #error is the error from error graph
        error = self.error_graph

        #optimizer is self.optimizer
        optimizer = self.optimizer(learning_rate=self.learning_rate_graph)

        return optimizer.minimize(error)