import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import math


class Highway(tf.Module): #from "Training very deep networks" paper
    def __init__(self, parameters):
        super(Highway,self).__init__()
        self.num_layers = parameters.num_highway_layers
        self.non_linear = [keras.layers.Dense(parameters.n_embed,input_shape=(parameters.n_embed,), activation=None) for i in range(self.num_layers)]
        self.linear = [keras.layers.Dense(parameters.n_embed,input_shape=(parameters.n_embed,), activation=None) for i in range(self.num_layers)]
        self.gate = [keras.layers.Dense(parameters.n_embed,input_shape=(parameters.n_embed,), activation=None) for i in range(self.num_layers)]
    
    def __call__(self, x):
        for layer in range(self.num_layers):
            transform_gate = keras.activations.sigmoid(self.linear[layer](x))
            carry_gate = 1-transform_gate
            non_linear = keras.activations.relu(self.non_linear[layer](x))
            linear = self.linear[layer](x)
            x = transform_gate*non_linear + carry_gate*linear
        return x


class Encoder(tf.Module): 
    def __init__(self,parameters):
        super(Encoder,self).__init__()
        self.highway = Highway(parameters)
        self.n_hidden_E = parameters.n_hidden_E
        self.n_layers_E = parameters.n_layers_E
        self.lstm = keras.layers.Bidirectional(keras.layers.LSTM(parameters.n_layers_E))

    def init_hidden(self,batch_size):
        h_0 = tf.zeros([2*self.n_layers_E, batch_size, self.n_hidden_E]) #initial state
        c_0 = tf.zeros([2*self.n_layers_E, batch_size, self.n_hidden_E])
        self.hidden = (h_0,c_0) 

    def __call__(self, x):
        batch_size, n_seq, n_embed = x.shape
        x = self.highway(x)
        self.init_hidden(batch_size)
        _, (self.hidden, _) = self.lstm(x,self.hidden)
        self.hidden = self.hidden.reshape([self.n_layers_E,2,batch_size,self.n_hidden_E])
        self.hidden = self.hidden[-1]
        e_hidden = tf.concat(list(self.hidden))
        return e_hidden
    
class Generator(tf.Module):
    def __init__(self, parameters):
        super(Generator,self).__init__()
        self.n_hidden_G = parameters.n_hidden_G
        self.n_layers_G = parameters.n_layers_G
        self.n_z = parameters.n_z #number of latent variables
        self.lstm = keras.layers.LSTM(parameters.num_layers)
        self.fc = keras.layers.Dense(parameters.n_vocab, input_shape=(parameters.n_hidden_G,), activation=None) #don't know what this layer does

    def init_hidden(self, batch_size):
        h_0 = tf.zeros([2*self.n_layers_G, batch_size, self.n_hidden_G])
        c_0 = tf.zeros([2*self.n_layers_G, batch_size, self.n_hidden_G]) #initialize covariance matrix
        self.hidden = (h_0,c_0) 

    def __call__(self, x, z, g_hidden = None):
        batch_size, n_seq, n_embed = x.shape
        z = tf.concat([z]*n_seq).reshape(batch_size, n_seq, self.n_z)
        x = tf.concat([x,z],axis = 2)

        if g_hidden is None:
            self.init_hidden(batch_size)
        else:
            self.hidden = g_hidden
        output, self.hidden = self.lstm(x, self.hidden)
        output = self.fc(output)

        return output, self.hidden
    
class VAE(tf.Module):
    def __init__(self, parameters):
        super(VAE, self).__init__()
        self.embedding = keras.layers.Embedding(parameters.n_vocab, parameters.n_embed)
        self.encoder = Encoder(parameters)
        self.hidden_to_mu = keras.layers.Dense(parameters.n_z, input_shape = (2*parameters.n_hidden_E,), activation = None)
        self.hidden_to_logvar = keras.layers.Dense(parameters.n_z, input_shape = (2*parameters.n_hidden_G,), activation = None)
        self.generator = Generator(parameters)
        self.n_z = parameters.n_z
    
    def __call__(self, x, G_inp, z=None, G_hidden=None):
        if z is None:
            batch_size, n_seq = x.shape
            x = self.embedding(x)
            E_hidden = self.encoder(x)
            mu = self.hidden_to_mu(E_hidden)
            log_var = self.hidden_to_logvar(E_hidden)
            z = tf.random.normal([batch_size,self.n_z])
            z = mu + tf.exp(0.5*log_var)*z
            kl = -0.5*sum(1+log_var-mu.pow(2)-log_var.exp()).mean()
        else:
            kl = None
        
        G_inp = self.embedding(G_inp)
        logit, G_hidden = self.generator(G_inp, z, G_hidden)
        return logit, G_hidden, kl




    
        
        

