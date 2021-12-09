import tensorflow as tf
import argparse
import math
import os
from vae import VAE
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_vocab', type=int, default=12000)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--n_hidden_G', type=int, default=512)
parser.add_argument('--n_layers_G', type=int, default=2)
parser.add_argument('--n_hidden_E', type=int, default=512)
parser.add_argument('--n_layers_E', type=int, default=1)
parser.add_argument('--n_z', type=int, default=100)
parser.add_argument('--word_dropout', type=float, default=0.5)
parser.add_argument('--rec_coef', type=float, default=7)
parser.add_argument('--lr', type=float, default=0.00001)
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


def generate_input(x, train):
    generator_input = x[:,0:x.shape[1]-1]
    if not train:
        return generator_input
    dropout_vals = np.random.rand(generator_input.shape(0),generator_input.shape(1))
    for i in range(len(generator_input)):
        for j in range(1,generator_input.shape[1]):
            if dropout_vals[i,j] < parameters.word_droput:
                generator_input[i,j] = parameters.unk_token
    return generator_input

def train_batch(x, generator_input, step, train=True):
    logit, _, kld = vae(x, generator_input, None, None)
    logit = logit.view(-1, opt.n_vocab)	                    #converting into shape (batch_size*(n_seq-1), n_vocab) to facilitate performing F.cross_entropy()
    x = x[:, 1:x.size(1)]	                                #target for generator should exclude first word of sequence
    x = x.contiguous().view(-1)	                            #converting into shape (batch_size*(n_seq-1),1) to facilitate performing F.cross_entropy()
    rec_loss = F.cross_entropy(logit, x)
    kld_coef = (math.tanh((step - 15000)/1000) + 1) / 2
    # kld_coef = min(1,step/(200000.0))
    loss = opt.rec_coef*rec_loss + kld_coef*kld
    if train==True:	                                    #skip below step if we are performing validation
        trainer_vae.zero_grad()
        loss.backward()
        trainer_vae.step()
    return rec_loss.item(), kld.item()


