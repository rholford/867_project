import tensorflow as tf
import argparse
import math
import os
from vae import VAE
import numpy as np
import tensorflow.keras as keras


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_vocab', type=int, default=12000)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--n_hidden_G', type=int, default=512)
parser.add_argument('--n_layers_G', type=int, default=2)
parser.add_argument('--n_hidden_E', type=int, default=512)
parser.add_argument('--n_layers_E', type=int, default=1)
parser.add_argument('--n_z', type=int, default=100)
parser.add_argument('--word_dropout', type=float, default=0.62) #from generating sentences paper
parser.add_argument('--rec_coef', type=float, default=7)
parser.add_argument('--learning_rate', type=float, default=0.00001)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--num_highway_layers', type=int, default=2)
parser.add_argument('--n_embed', type=int, default=300)
parser.add_argument('--out_num', type=int, default=30000)
parser.add_argument('--unk_token', type=str, default="<unk>")
parser.add_argument('--pad_token', type=str, default="<pad>")
parser.add_argument('--start_token', type=str, default="<sos>")
parser.add_argument('--end_token', type=str, default="<eos>")
parser.add_argument('--dataset', type=str, default="reddit")
parser.add_argument('--training', action='store_true')
parser.add_argument('--resume_training', action='store_true')

parameters = parser.parse_args()

vae = VAE(parameters)



def generate_input(x, train):
    generator_input = x[:,0:x.shape[1]-1]
    if not train:
        return generator_input
    dropout_vals = np.random.rand(generator_input.shape[0],generator_input.shape[1])
    for i in range(len(generator_input)):
        for j in range(1,generator_input.shape[1]): #need to map vocab to numerical values
            if dropout_vals[i,j] < parameters.word_droput:
                generator_input[i,j] = parameters.unk_token
    return generator_input


def train_batch(x, generator_input, step, train=True):
    logit,_,kl = VAE(x, generator_input, z=None, G_hidden= None)
    logit = tf.reshape(logit,[parameters.batch_size*(parameters.n_seq-1),parameters.n_vocab])
    x = x[:,1:x.shape[1]]
    x = tf.reshape(x,[-1])
    recreation_loss = tf.nn.softmax_cross_entropy_with_logits(x, logit)
    kl_coef = 0.5*math.tanh((step-15000)/1000+1) #KL annealing technique, ask author why set to this
    loss = kl_coef*kl+ parameters.rec_coef*recreation_loss
    

def training():
    start_epoch = 0
    step = 0
    if parameters.resume_training:
        step = checkpoint['step']
        start_epoch = checkpoint['epoch']
        for e in range(start_epoch,parameters.epochs):
            #vae.fit()
            train_rec_loss = []
            train_kl_loss = []
            for batch in train_iter:
                x = batch.text
                generator_input = generate_input(x,train=True)
                rec_loss, kl_loss = train_batch(x, generator_input, step, train=True)
                train_rec_loss.append(rec_loss)
                train_kl_loss.append(kl_loss)
                step = step + 1
                
                #vae.evaluate()
                valid_rec_loss = []
                valid_kl_loss = []
                for batch in val_iter:
                    x = batch.text
                    generator_input = generate_input(x,train=False)
                    tf.stop_gradient(parameters)
                    rec_loss, kl_loss = train_batch(x, generator_input, step, train=False)
                    valid_rec_loss.append(rec_loss)
                    valid_kl_loss.append(kl_loss)

                rec_loss_train = np.mean(train_rec_loss)
                kl_loss_train = np.mean(train_kl_loss)
                rec_loss_valid = np.mean(valid_rec_loss)
                kl_loss_valid = np.mean(valid_kl_loss)

                print("No.", e, "T_rec:", '%.2f' % rec_loss_train, "T_kld:", '%.2f' % kl_loss_train, "V_rec:", '%.2f' % rec_loss_valid, "V_kld:", '%.2f' % kl_loss_valid)

                if e>=50 and e%10==0:
                    print ("Save model " + str(e) + "...")
                    #save model
                    #generate_sentences(5)




