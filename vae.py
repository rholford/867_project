import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.layers.recurrent_v2 import lstm_with_backend_selection







class Highway(keras.layers.Layer):
    def __init__(self, opt):
        super(Highway,self).__init__()
        self.n_layers = opt.n_highway_layers
        self.non_linear = [keras.layers.Dense(opt.n_embed,input_shape=(opt.n_embed,), activation=None) for i in range(self.n_layers)]
        self.linear = [keras.layers.Dense(opt.n_embed,input_shape=(opt.n_embed,), activation=None) for i in range(self.n_layers)]
        self.gate = [keras.layers.Dense(opt.n_embed,input_shape=(opt.n_embed,), activation=None) for i in range(self.n_layers)]
    
    def forward(self, x):
        for layer in range(self.n_layers):
            gate = keras.activations.sigmoid(self.linear[layer](x))
            non_linear = keras.activations.relu(self.non_linear[layer](x))
            linear = self.linear[layer](x)
            x = gate*non_linear + (1-gate)*linear
        return x


class Encoder(keras.layers.Layer):
    def __init__(self,opt):
        super(Encoder,self).__init__()
        self.highway = Highway(opt)
        self.n_hidden_E = opt.n_hidden_E
        self.n_layers_E = opt.n_layers_E
        #self.lstm = keras.layers.Bidirectional(keras.layers.LSTM(opt.n_layers_E))

    def init_hidden(self,batch_size):
        h_0 = tf.zeros([2*self.n_layers_E, batch_size, self.n_hidden_E])
        c_0 = tf.zeros([2*self.n_layers_E, batch_size, self.n_hidden_E])
        self.hidden = (h_0,c_0) #may need get_cuda function, idk how it works in tensorflow

    def forward(self, x):
        batch_size, n_seq, n_embed = tf.size(x)
        x = self.highway(x)
        self.init_hidden(batch_size)
        _, (self.hidden, _) = self.lstm(x,self.hidden)
        self.hidden = self.hidden.reshape(self.n_layers_E,2,batch_size,self.n_hidden_E)
        self.hidden = self.hidden[-1]
        e_hidden = tf.concat(list(self.hidden))
        return e_hidden
    
class Generator(keras.layers.Layer):
    def __init__(self, opt):
        super(Generator,self).__init__()
        self.n_hidden_G = opt.n_hidden_G
        self.n_layers_G = opt.n_layers_G
        self.n_z = opt.n_z
        #self.lstm needed, again idk how to in tensorflow
        self.fc = keras.layers.Dense(opt.n_vocab, input_shape=(opt.n_hidden_G,), activation=None) #don't know what this layer does

    def init_hidden(self, batch_size):
        h_0 = tf.zeros([2*self.n_layers_G, batch_size, self.n_hidden_G])
        c_0 = tf.zeros([2*self.n_layers_G, batch_size, self.n_hidden_G])
        self.hidden = (h_0,c_0) #may need get_cuda function, idk how it works in tensorflow

    def forward(self, x, z, g_hidden = None):
        batch_size, n_seq, n_embed = tf.size(x)
        z = tf.concat([z]*n_seq).reshape(batch_size, n_seq, self.n_z)
        x = tf.concat([x,z],axis = 2)

        if g_hidden is None:
            self.init_hidden(batch_size)
        else:
            self.hidden = g_hidden
        output, self.hidden = self.lstm(x, self.hidden)
        output = self.fc(output)

        return output, self.hidden
    
class VAE(keras.layers.Layer):
    def __init__(self, opt):
        super(VAE, self).__init__()
        self.embedding = keras.layers.Embedding(opt.n_vocab, opt.n_embed)
        self.encoder = Encoder(opt)
        self.hidden_to_mu = keras.layers.Dense(opt.n_z, input_shape = (2*opt.n_hidden_E,), activation = None)
        self.hidden_to_logvar = keras.layers.Dense(opt.n_z, input_shape = (2*opt.n_hidden_G,), activation = None)
        self.generator = Generator(opt)
        self.n_z = opt.n_z
