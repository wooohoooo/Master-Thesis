
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
#tf.reset_default_graph()
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
from os import system


# https://danijar.com/structuring-your-tensorflow-models/

# In[2]:


import functools

def lazy_property(function):
    """
    Decorator makes sure nodes are only appended if they dont already exist
    """
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


# In[3]:




class EnsembleNetwork(object):
    def __init__(self,
                 num_neurons=[10,10,10],
                 num_features=1,
                 learning_rate=0.001,
                 activations = None, #[tf.nn.tanh,tf.nn.relu,tf.sigmoid]
                 dropout_layers = None, #[True,False,True]
                 initialisation_scheme= None, #[tf.random_normal,tf.random_normal,tf.random_normal]
                 optimizer = None, #defaults to GradiendDescentOptimizer,
                 num_epochs = None, #defaults to 1,
                 seed = None
                 ):
        
        #necessary parameters
        self.num_neurons = num_neurons
        self.num_layers = len(num_neurons)
        self.num_features = num_features
        self.learning_rate = learning_rate
        
        #optional parameters
        self.optimizer = optimizer or tf.train.GradientDescentOptimizer
        self.activations = activations or [tf.nn.tanh]*self.num_layers
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
        self.X = tf.placeholder(tf.float32,(None,self.num_features))
        self.y = tf.placeholder(tf.float32,(None,1))#regression = 1
        
        #lists for storage
        self.w_list = []
        self.b_list = []
        
        #add input x first weights
        self.w_list.append(tf.Variable(self.initialisation_scheme([self.num_features,self.num_neurons[0]]),name='w_0')) #first Matrix
        
        #for each layer over 0 add a n x m matrix and a bias term
        for i,num_neuron in enumerate(self.num_neurons[1:]):
            n_inputs = self.num_neurons[i] #for first hidden layer 3
            n_outputs = self.num_neurons[i+1] #for first hidden layer 5
                                                    
            self.w_list.append(tf.Variable(self.initialisation_scheme([n_inputs,n_outputs]),name='w_'+str(i)))
            self.b_list.append(tf.Variable(tf.ones(shape=[n_inputs]),name='b_'+str(i)))
            
        #add last layer m  x 1 for output
        self.w_list.append(tf.Variable(self.initialisation_scheme([self.num_neurons[-1],1]),name='w_-1'))#this is a regression
        self.b_list.append(tf.Variable(tf.ones(shape=[self.num_neurons[-1]]),name='b_'+str(len(self.num_neurons)+1)))

    @lazy_property 
    def predict_graph(self):
        #set layer_input to input
        layer_input = self.X
        
        #for each layer do
        for i,w in enumerate(self.w_list):
            
            #z = input x Weights
            a = tf.matmul(layer_input,w,name='matmul_'+str(i)) 
            
            #z + bias
            if i < self.num_layers:
                bias = self.b_list[i]
                a = tf.add(a,bias)
                
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
        error = tf.losses.mean_squared_error(self.y,y_hat)
        
        return error
    
    @lazy_property
    def train_graph(self):
        
        #error is the error from error graph
        error = self.error_graph
        
        #optimizer is self.optimizer
        optimizer = self.optimizer(learning_rate = self.learning_rate)
        
        return optimizer.minimize(error)
    

    def train(self,X,y):
        for epoch in range(self.num_epochs):

            self.session.run(self.train_graph,feed_dict={self.X:X,self.y:y})
        
    def predict(self,X):
        return self.session.run(self.predict_graph,feed_dict={self.X:X})


    def kill(self):
        self.session.close()
            


class EnsembleNetworkWeird(object):
    def __init__(self,
                 num_neurons=[10,10,10],
                 num_features=1,
                 learning_rate=0.001,
                 activations = None, #[tf.nn.tanh,tf.nn.relu,tf.sigmoid]
                 dropout_layers = None, #[True,False,True]
                 initialisation_scheme= None, #[tf.random_normal,tf.random_normal,tf.random_normal]
                 optimizer = None, #defaults to GradiendDescentOptimizer,
                 num_epochs = None, #defaults to 1,
                 seed = None
                 ):
        
        #necessary parameters
        self.num_neurons = num_neurons
        self.num_layers = len(num_neurons)
        self.num_features = num_features
        self.learning_rate = learning_rate
        
        #optional parameters
        self.optimizer = optimizer or tf.train.GradientDescentOptimizer
        self.activations = activations or [tf.nn.tanh]*self.num_layers
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
        self.X = tf.placeholder(tf.float32,(None,self.num_features))
        self.y = tf.placeholder(tf.float32,(None,1))#regression = 1
        
        #lists for storage
        self.w_list = []
        self.b_list = []
        
        #add input x first weights
        self.w_list.append(tf.Variable(self.initialisation_scheme([self.num_features,self.num_neurons[0]]),name='w_0')) #first Matrix
        
        #for each layer over 0 add a n x m matrix and a bias term
        for i,num_neuron in enumerate(self.num_neurons[1:]):
            n_inputs = self.num_neurons[i] #for first hidden layer 3
            n_outputs = self.num_neurons[i+1] #for first hidden layer 5
                                                    
            self.w_list.append(tf.Variable(self.initialisation_scheme([n_inputs,n_outputs]),name='w_'+str(i)))
            self.b_list.append(tf.Variable(tf.ones(shape=[n_inputs]),name='b_'+str(i)))
            
        #add last layer m  x 1 for output
        self.p_w = tf.Variable(self.initialisation_scheme([self.num_neurons[-1],1]),name='w_-1')#this is a regression
        self.std_w = tf.Variable(self.initialisation_scheme([self.num_neurons[-1],1]),name='w_std')#this is a regression

        self.b_list.append(tf.Variable(tf.ones(shape=[self.num_neurons[-1]]),name='b_'+str(len(self.num_neurons)+1)))

    @lazy_property 
    def predict_graph(self):
        #set layer_input to input
        layer_input = self.X
        
        #for each layer do
        for i,w in enumerate(self.w_list):
            
            #z = input x Weights
            a = tf.matmul(layer_input,w,name='matmul_'+str(i)) 
            
            #z + bias
            #if i < self.num_layers:
            bias = self.b_list[i]
            a = tf.add(a,bias)
            
            #a = sigma(z) if not last layer and regression
            #if i < self.num_layers:    

            a = self.activations[i](a)
            
            #set layer input to a for next cycle
            layer_input = a
            
        return a

    @lazy_property
    def p_graph(self):
        l_in = self.predict_graph
        return tf.matmul(l_in,self.p_w)


    @lazy_property
    def std_graph(self):
        l_in = self.predict_graph
        return tf.nn.softplus(tf.matmul(l_in,self.std_w))
    
    
    @lazy_property
    def error_graph(self):
        
        #y_hat is a // output of prediction graph
        y_hat = self.p_graph
        std_hat = tf.maximum(self.std_graph,0.001)#tf.square(self.std_graph)
        #s_hat = self.predict_graph[1]
        
        #error is mean squared error of placehilder y and prediction
        #error = tf.losses.mean_squared_error(self.y,y_hat)
        #first_term = tf.div(tf.square(std_hat),2.0)
        #second_term = tf.div(tf.square(tf.subtract(self.y,y_hat)), tf.multiply(2.0,std_hat) )
        first_term = tf.log(tf.div(std_hat,2.0))
        second_term = tf.div(tf.square(tf.subtract(self.y,y_hat)), tf.multiply(2.0,std_hat) )
        #error = -tf.log(tf.add(first_term,second_term))
        error = tf.add(tf.add(first_term,second_term),1.0)
        #error = 
        
        return error
    
    @lazy_property
    def train_graph(self):
        
        #error is the error from error graph
        error = self.error_graph
        
        #optimizer is self.optimizer
        optimizer = self.optimizer(learning_rate = self.learning_rate)
        
        return optimizer.minimize(error)
    

    def train(self,X,y):
        for epoch in range(self.num_epochs):
            for batch_X,batch_y in zip(X,y):
                batch_X = np.expand_dims(batch_X,1)
                batch_y = np.expand_dims(batch_y,1)
                self.session.run(self.train_graph,feed_dict={self.X:batch_X,self.y:batch_y})
        
    def predict(self,X):
        return self.session.run(self.p_graph,feed_dict={self.X:X})

    def predict_var(self,X):
        return self.session.run(self.std_graph,feed_dict={self.X:X})


    def kill(self):
        self.session.close()
            


class LearningrateNetwork(object):
    def __init__(self,
                 num_neurons=[10,10,10],
                 num_features=1,
                 learning_rate=0.001,
                 activations = None, #[tf.nn.tanh,tf.nn.relu,tf.sigmoid]
                 dropout_layers = None, #[True,False,True]
                 initialisation_scheme= None, #[tf.random_normal,tf.random_normal,tf.random_normal]
                 optimizer = None, #defaults to GradiendDescentOptimizer,
                 num_epochs = None, #defaults to 1,
                 seed = None
                 ):
        
        #necessary parameters
        self.num_neurons = num_neurons
        self.num_layers = len(num_neurons)
        self.num_features = num_features
        
        #optional parameters
        self.optimizer = optimizer or tf.train.GradientDescentOptimizer
        self.activations = activations or [tf.nn.tanh]*self.num_layers
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
        self.X = tf.placeholder(tf.float32,(None,self.num_features))
        self.y = tf.placeholder(tf.float32,(None,1))#regression = 1
        
        #learning rate Variable
        self.learning_rate = tf.Variable(self.learning_rate_init)

        #lists for storage
        self.w_list = []
        self.b_list = []
        
        #add input x first weights
        self.w_list.append(tf.Variable(self.initialisation_scheme([self.num_features,self.num_neurons[0]]),name='w_0')) #first Matrix
        
        #for each layer over 0 add a n x m matrix and a bias term
        for i,num_neuron in enumerate(self.num_neurons[1:]):
            n_inputs = self.num_neurons[i] #for first hidden layer 3
            n_outputs = self.num_neurons[i+1] #for first hidden layer 5
                                                    
            self.w_list.append(tf.Variable(self.initialisation_scheme([n_inputs,n_outputs]),name='w_'+str(i)))
            self.b_list.append(tf.Variable(tf.ones(shape=[n_inputs]),name='b_'+str(i)))
            
        #add last layer m  x 1 for output
        self.p_w = tf.Variable(self.initialisation_scheme([self.num_neurons[-1],1]),name='w_-1')#this is a regression
        self.std_w = tf.Variable(self.initialisation_scheme([self.num_neurons[-1],1]),name='w_std')#this is a regression

        self.b_list.append(tf.Variable(tf.ones(shape=[self.num_neurons[-1]]),name='b_'+str(len(self.num_neurons)+1)))

    @lazy_property 
    def predict_graph(self):
        #set layer_input to input
        layer_input = self.X
        
        #for each layer do
        for i,w in enumerate(self.w_list):
            
            #z = input x Weights
            a = tf.matmul(layer_input,w,name='matmul_'+str(i)) 
            
            #z + bias
            #if i < self.num_layers:
            bias = self.b_list[i]
            a = tf.add(a,bias)
            
            #a = sigma(z) if not last layer and regression
            #if i < self.num_layers:    

            a = self.activations[i](a)
            
            #set layer input to a for next cycle
            layer_input = a
            
        return a

    @lazy_property
    def p_graph(self):
        l_in = self.predict_graph
        return tf.matmul(l_in,self.p_w)


    @lazy_property
    def std_graph(self):
        l_in = self.predict_graph
        return tf.nn.softplus(tf.matmul(l_in,self.std_w))
    
    
    @lazy_property
    def error_graph(self):
        
        #y_hat is a // output of prediction graph
        y_hat = self.p_graph
        std_hat = tf.maximum(self.std_graph,0.001)#tf.square(self.std_graph)
        #s_hat = self.predict_graph[1]
        
        #error is mean squared error of placehilder y and prediction
        #error = tf.losses.mean_squared_error(self.y,y_hat)
        #first_term = tf.div(tf.square(std_hat),2.0)
        #second_term = tf.div(tf.square(tf.subtract(self.y,y_hat)), tf.multiply(2.0,std_hat) )
        first_term = tf.log(tf.div(std_hat,2.0))
        second_term = tf.div(tf.square(tf.subtract(self.y,y_hat)), tf.multiply(2.0,std_hat) )
        #error = -tf.log(tf.add(first_term,second_term))
        error = tf.add(tf.add(first_term,second_term),1.0)
        #error = 
        
        return error

    @lazy_property
    def learning_rate_graph(self):
        learning_rate = tf.squeeze(tf.multiply(self.learning_rate,tf.maximum(tf.sqrt(self.std_graph),0.05)))
        return learning_rate

    @lazy_property
    def train_graph(self):
        
        #error is the error from error graph
        error = self.error_graph
        
        #optimizer is self.optimizer
        optimizer = self.optimizer(learning_rate = self.learning_rate_graph)
        
        return optimizer.minimize(error)
    

    def train(self,X,y):
        #loss_list =[]
        for epoch in range(self.num_epochs):
            #avg_loss_list = []
            for batch_X,batch_y in zip(X,y):

                batch_X = np.expand_dims(batch_X,1)
                #if epoch%10 ==0:
                #    avg_loss_list.append(self.get_loss(batch_X))
                batch_y = np.expand_dims(batch_y,1)
                self.session.run(self.train_graph,feed_dict={self.X:batch_X,self.y:batch_y})
            #loss_list.append(np.mean(avg_loss_list))
        #return loss_list
        
    def predict(self,X):
        #if return_loss:
        #    return {'predictions':self.session.run(self.p_graph,feed_dict={self.X:X}),
        #    'loss':self.session.run(self.error_graph,feed_dict={self.X:X})}

        return self.session.run(self.p_graph,feed_dict={self.X:X})

    def predict_var(self,X):
        return self.session.run(self.std_graph,feed_dict={self.X:X})

    #def get_loss(self,X):
    #    return self.session.run(self.error_graph,feed_dict={self.X:X})


    def kill(self):
        self.session.close()
            

# # Data

# In[4]:


def make_y(X,noise=True):
    func = 5 * np.sin(X) + 10 
    if noise:
        noise = np.random.normal(0,4,size=X.shape)
        return func + noise
    return func


def expand_array_dims(array):
    new_array = [np.expand_dims(np.array(x),0) for x in array]
    #new_array = [np.expand_dims(x,1) for x in new_array]

    return new_array
 

class EnsembleParent(object):
    """
    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/05_Ensemble_Learning.ipynb
    """
    def __init__(self,estimator_stats = None,num_epochs=10):
        pass
        
    def train(self,X,y):
        pass
            
    def predict(self,X,return_samples=False):
        pass
            
        
    def plot(self,X,y,plot_samples=False,original_func=None,sorted_index=None):
        preds_dict = self.predict(X,True)
        y_hats = np.array([mean[0] for mean in preds_dict['means']])
        y_stds = np.array([std[0] for std in preds_dict['stds']])
        samples = preds_dict['samples']
        X_plot = [x[0] for x in np.array(X)]#[sorted_index]]
        
        #print(y_hats.shape)
        #print(y_stds.shape)
        #print(y.shape)
        #print(np.array(X).shape)
        #X_plot = X_plot[sorted_index]
        y = np.array(y)#[sorted_index]
        y_hats = y_hats#[sorted_index]
        y_stds = y_stds#[sorted_index]

        plt.plot(X_plot,y,'kx',label='Data',alpha = 0.4)
 
        plt.plot(X_plot,y_hats,label='predictive mean')
        plt.fill_between(X_plot,y_hats+y_stds,y_hats,alpha=.3,color='b')
        plt.fill_between(X_plot,y_hats-y_stds,y_hats,alpha=.3,color='b')

        if plot_samples:
            for i,sample in enumerate(samples):
                
                plt.plot(X_plot,sample,label='sample {}'.format(i),alpha=max(1/len(samples),0.3))
        if original_func:
            y_original = original_func(X_plot,noise=False)
            plt.plot(X,y_original,label='generating model')
        plt.legend()
        
        
        
        
    


# # Ensemble

# In[9]:


class EnsembleEstimator(EnsembleParent):
    """
    https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/05_Ensemble_Learning.ipynb
    """
    def __init__(self,estimator_stats = None,num_epochs=10):
        
        default_ensemble = [{'num_neurons':[2,2,2],'num_epochs':num_epochs},
                            {'num_neurons':[10,2],'num_epochs':num_epochs},
                            {'num_neurons':[5,5],'num_epochs':num_epochs}]
        
        
        self.estimator_stats = estimator_stats or default_ensemble
        self.estimator_list = [EnsembleNetwork(**x) for x in self.estimator_stats]
        
    def train(self,X,y):
        '''This is where we build in the Online Bootstrap'''
        for estimator in self.estimator_list:
            estimator.train(X,y)
        system('say training  complete')

            
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
        system('say prediction complete')

        return return_dict
            
        
    def plot_old(self,X,y,plot_samples=False,original_func=None):
        preds_dict = self.predict(X,True)
        y_hats = np.array([mean[0] for mean in preds_dict['means']])
        y_stds = np.array([std[0] for std in preds_dict['stds']])
        samples = preds_dict['samples']
        X_plot = [x[0] for x in X]
        
        #print(y_hats.shape)
        #print(y_stds.shape)
        #print(y.shape)
        #print(np.array(X).shape)
        plt.plot(X_plot,y,'kx',label='Data')
 
        plt.plot(X_plot,y_hats,label='predictive mean',alpha=0.2)
        plt.fill_between(X_plot,y_hats+y_stds,y_hats,alpha=.3,color='b')
        plt.fill_between(X_plot,y_hats-y_stds,y_hats,alpha=.3,color='b')

        if plot_samples:
            for i,sample in enumerate(samples):
                
                plt.plot(X_plot,sample,label='sample {}'.format(i))
        if original_func:
            y_original = original_func(X_plot,noise=False)
            plt.plot(X,y_original,label='generating model')
        plt.legend()
        
        
        
        
    


# In[10]:



class BootstrapEstimator(EnsembleParent):
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
                print(len(dataset))
                self.dataset_list.append({'X':dataset,'y':labels})
                old_mask = mask
            

            
            
            
            
        for estimator,dataset in zip(self.estimator_list,self.dataset_list):
            estimator.train(dataset['X'],dataset['y'])
        system('say training complete')
            
            
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
        system('say prediction complete')

        return return_dict
            
        
    def plot_old(self,X,y,plot_samples=False,original_func=None):
        preds_dict = self.predict(X,True)
        y_hats = np.array([mean[0] for mean in preds_dict['means']])
        y_stds = np.array([std[0] for std in preds_dict['stds']])
        samples = preds_dict['samples']
        X_plot = [x[0] for x in X]
        
        #print(y_hats.shape)
        #print(y_stds.shape)
        #print(y.shape)
        #print(np.array(X).shape)
        plt.plot(X_plot,y,'kx',label='Data')

        plt.plot(X_plot,y_hats,label='predictive mean')
        plt.fill_between(X_plot,y_hats+y_stds,y_hats,alpha=.3,color='b')
        plt.fill_between(X_plot,y_hats-y_stds,y_hats,alpha=.3,color='b')

        if plot_samples:
            for i,sample in enumerate(samples):
                
                plt.plot(X_plot,sample,label='sample {}'.format(i))
        if original_func:
            y_original = original_func(X_plot,noise=False)
            plt.plot(X,y_original,label='generating model')
        plt.legend()
        
        
        
        
    






from IPython.display import clear_output, Image, display, HTML

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


# In[22]:


#show_graph(ensemble.estimator_list[0].g.as_graph_def())


# In[ ]:




