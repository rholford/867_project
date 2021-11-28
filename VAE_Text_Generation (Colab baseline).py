""" VAE for Text Generation
This is for Module 1: Candidates Generation.
Usage: python VAE_Text_Generation.py --dataset reddit
"""
import argparse
import math
import os
import numpy as np
import torch as T
import torch.nn.functional as F
from tqdm import tqdm
from utility.VAE_Text_Generation.dataset import get_iterators
from utility.VAE_Text_Generation.helper_functions import get_cuda
#from utility.VAE_Text_Generation.model import VAE
from utility.VAE_Text_Generation.vae import VAE, Encoder #added encoder
import tensorflow as tf 
import tensorflow as keras #added?
import tensorflow_addons as tfa #added
from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda
from keras.layers import LSTM, RepeatVector, TimeDistributed, Layer, Activation, Dropout
#imports added 
from keras.models import Model
from keras import backend
from tensorflow.keras.models import Sequential

def zero_loss(y_hat):
    return backend.zeros_like(y_hat)

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
parser.add_argument('--n_highway_layers', type=int, default=2)
parser.add_argument('--n_embed', type=int, default=300)
parser.add_argument('--out_num', type=int, default=30000)
parser.add_argument('--unk_token', type=str, default="<unk>")
parser.add_argument('--pad_token', type=str, default="<pad>")
parser.add_argument('--start_token', type=str, default="<sos>")
parser.add_argument('--end_token', type=str, default="<eos>")
parser.add_argument('--dataset', type=str, default="reddit")
parser.add_argument('--training', action='store_true')
parser.add_argument('--resume_training', action='store_true')



#opt = parser.parse_args()
#print(opt)
#switching to param for name issue?
#Ever instance outside of the "new code" block was switched from 'opt' to 'parameters'
parameters = parser.parse_args()
save_path = "tmp/saved_VAE_models/" + parameters.dataset + ".tar"
print(save_path)
if not os.path.exists("tmp/saved_VAE_models"):
    os.makedirs("tmp/saved_VAE_models")
os.environ["CUDA_VISIBLE_DEVICES"] = str(parameters.gpu)
print ("testing 1")
candidates_path = parameters.dataset + '_for_VAE.txt'
train_iter, val_iter, vocab = get_iterators(parameters, path='./data/', fname=candidates_path)
print ("testing 2")
parameters.n_vocab = len(vocab)
print (type(vocab))
print (len(vocab))
#print (train_iter)
#print (type(val_iter))

#New code added below   
#parameters = parser.parse_args()
#vocab = tf.ones([12000,300])
print ("testing 3")
vocab = tf.ones_like(vocab) #ones to ones_like
print ("testing 4")
print((vocab))
x = Input(batch_shape=(None,15))
x_embed = Embedding(parameters.n_vocab,parameters.n_embed,weights=[vocab],input_length=15,trainable=False)(x)
h = Bidirectional(LSTM(parameters.n_hidden_G,return_sequences=False,recurrent_dropout=0.2),merge_mode="concat")(x_embed)
h = Dropout(0.2)(h)
mu = Dense(parameters.n_z)(h)
log_var = Dense(parameters.n_z)(h)

def sample(args):
    print ("entered sample func")
    mu, log_var = args
    eps = tf.random.normal(shape=(parameters.batch_size,parameters.n_z),mean=0,stddev=1)
    return mu + tf.exp(0.5*log_var)*eps

z = Lambda(sample,output_shape=(parameters.n_z,))([mu,log_var])
repeat_vector = RepeatVector(15)
decoder_h = LSTM(parameters.n_hidden_E,return_sequences=True,recurrent_dropout=0.2)
decoder_mu = TimeDistributed(Dense(parameters.n_vocab,activation='linear'))
decoded_h = decoder_h(repeat_vector(z))
decoded_mu = decoder_mu(decoded_h)
print ("functioning")
logits = tf.constant(np.random.randn(parameters.batch_size, 15, parameters.n_vocab), tf.float32)
targets = tf.constant(np.random.randint(parameters.n_vocab, size=(parameters.batch_size, 15)), tf.int32)
proj_w = tf.constant(np.random.randn(parameters.n_vocab, parameters.n_vocab), tf.float32)
proj_b = tf.constant(np.zeros(parameters.n_vocab), tf.float32)

def sample_loss(labels, logits):
    print ("sample loss")
    labels = labels.reshape(tf.cast(labels,tf.int64),[-1,1])
    logits = tf.cast(logits, tf.float32)
    return tf.cast(tf.nn_sampled_softmax_loss(
        proj_w, proj_b, labels, logits, num_sampled=500,num_classes = parameters.n_vocab
    ),tf.int32)

softmax_loss = sample_loss

class VAELayer(Layer):
    def __init__(self, **kwargs):
        super(VAELayer,self).__init__(**kwargs)
        self.target_weights = tf.constant(np.ones((parameters.batch_size, 15)),tf.float32) 

    def vae_loss(self, x, decoded_mu):
        labels = tf.cast(x, tf.int32)
        recreation_loss = backend.sum(tfa.seq2seq.sequence_loss(
            decoded_mu,labels,weights=self.target_weights,average_across_timesteps=False,
            average_across_batch=False),axis=-1)

        kl_loss = -0.5*backend.sum(1+log_var-backend.square(mu)-backend.exp(log_var))
        return backend.mean(kl_loss + recreation_loss)

    def call(self, inputs):
        print ("call")
        x = inputs[0]
        decoded_mu = inputs[1]
        loss = self.vae_loss(x,decoded_mu)
        self.add_loss(loss,inputs=inputs)
        return backend.zeros_like(x)

loss_layer = VAELayer()([x,decoded_mu])
vae = Model(x,[loss_layer])
opt = keras.optimizers.Adam(lr=parameters.lr)
vae.compile(optimizer=opt,loss=[zero_loss])

#end of new code






if parameters.training:
    vae = VAE(parameters)
    vae.embedding.embeddings_initializer = tf.constant(vocab.vectors) 
    #vae.embedding.weight.data.copy_(vocab.vectors)              #Intialize trainable embeddings with pretrained glove vectors
    vae = get_cuda(vae)
    #trainer_vae = T.optim.Adam(vae.parameters(), lr=opt.lr)
else:
    checkpoint = T.load(save_path)
    vae = checkpoint['vae_dict']
    trainer_vae = checkpoint['vae_trainer']
    if 'parameters' in checkpoint:
        parameters_old = checkpoint['parameters']
        print(parameters_old)


def create_generator_input(x, train):
    G_inp = x[:, 0:x.size(1)-1].clone()	                    #input for generator should exclude last word of sequence
    if train == False:
        return G_inp
    r = np.random.rand(G_inp.size(0), G_inp.size(1))        #Perform word_dropout according to random values (r) generated for each word
    for i in range(len(G_inp)):
        for j in range(1, G_inp.size(1)):
            if r[i, j] < parameters.word_dropout and G_inp[i, j] not in [vocab.stoi[parameters.pad_token], vocab.stoi[parameters.end_token]]:
                G_inp[i, j] = vocab.stoi[parameters.unk_token]
    return G_inp


def train_batch(x, G_inp, step, train=True):
    logit, _, kld = vae(x, G_inp, None, None)
    logit = logit.view(-1, parameters.n_vocab)	                    #converting into shape (batch_size*(n_seq-1), n_vocab) to facilitate performing F.cross_entropy()
    x = x[:, 1:x.size(1)]	                                #target for generator should exclude first word of sequence
    x = x.contiguous().view(-1)	                            #converting into shape (batch_size*(n_seq-1),1) to facilitate performing F.cross_entropy()
    rec_loss = F.cross_entropy(logit, x)
    kld_coef = (math.tanh((step - 15000)/1000) + 1) / 2
    # kld_coef = min(1,step/(200000.0))
    loss = parameters.rec_coef*rec_loss + kld_coef*kld
    if train==True:	                                    #skip below step if we are performing validation
        trainer_vae.zero_grad()
        loss.backward()
        trainer_vae.step()
    return rec_loss.item(), kld.item()


# def load_model_from_checkpoint():
    # global vae, trainer_vae
    # checkpoint = T.load(save_path)
    # vae.load_state_dict(checkpoint['vae_dict'])
    # trainer_vae.load_state_dict(checkpoint['vae_trainer'])
    # return checkpoint['step'], checkpoint['epoch']

'''
#old
def training():
    start_epoch = step = 0
    if opt.resume_training:
        step, start_epoch = checkpoint['step'], checkpoint['epoch']
    for epoch in range(start_epoch, opt.epochs):
        vae.train()
        train_rec_loss = []
        train_kl_loss = []
        for batch in train_iter:
            x = get_cuda(batch.text) 	                                #Used as encoder input as well as target output for generator
            G_inp = create_generator_input(x, train=True)
            rec_loss, kl_loss = train_batch(x, G_inp, step, train=True)
            train_rec_loss.append(rec_loss)
            train_kl_loss.append(kl_loss)
            step += 1

        vae.eval()
        valid_rec_loss = []
        valid_kl_loss = []
        for batch in val_iter:
            x = get_cuda(batch.text)
            G_inp = create_generator_input(x, train=False)
            with T.autograd.no_grad():
                rec_loss, kl_loss = train_batch(x, G_inp, step, train=False)
            valid_rec_loss.append(rec_loss)
            valid_kl_loss.append(kl_loss)

        train_rec_loss = np.mean(train_rec_loss)
        train_kl_loss = np.mean(train_kl_loss)
        valid_rec_loss = np.mean(valid_rec_loss)
        valid_kl_loss = np.mean(valid_kl_loss)

        print("No.", epoch, "T_rec:", '%.2f' % train_rec_loss, "T_kld:", '%.2f' % train_kl_loss, "V_rec:", '%.2f' % valid_rec_loss, "V_kld:", '%.2f' % valid_kl_loss)
        if epoch >= 50 and epoch % 10 == 0:
            print('save model ' + str(epoch) + '...')
            T.save({'epoch': epoch + 1, 'vae_dict': vae, 'vae_trainer': trainer_vae, 'step': step, 'opt': opt}, save_path)
            generate_sentences(5)
'''

#from github -test, swapped parameters to opt+
#generate_input => generate_sentences (whoops.create_generator_input?)
def training():
    print ("entered training func")
    start_epoch = 0
    step = 0
    if opt.resume_training:
        step = checkpoint['step']
        start_epoch = checkpoint['epoch']
        for e in range(start_epoch, parameters.epochs):
            #vae.fit()
            train_rec_loss = []
            train_kl_loss = []
            for batch in train_iter:
                x = batch.text
                generator_input = create_generator_input(x,train=True)
                rec_loss, kl_loss = train_batch(x, generator_input, step, train=True)
                train_rec_loss.append(rec_loss)
                train_kl_loss.append(kl_loss)
                step = step + 1
                
                #vae.evaluate()
                valid_rec_loss = []
                valid_kl_loss = []
                for batch in val_iter:
                    x = batch.text
                    generator_input = create_generator_input(x,train=False)
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

def generate_sentences(n_examples, save=0):
    #vae.eval() #commented out? swapped? idk
    #vae.evaluate()
    out = []
    for i in tqdm(range(n_examples)):
        z = get_cuda(T.randn([1, vae.n_z]))
        h_0 = get_cuda(T.zeros(vae.generator.n_layers_G, 1, vae.generator.n_hidden_G))
        c_0 = get_cuda(T.zeros(vae.generator.n_layers_G, 1, vae.generator.n_hidden_G))
        G_hidden = (h_0, c_0)
        G_inp = T.LongTensor(1, 1).fill_(vocab.stoi[opt.start_token])
        G_inp = get_cuda(G_inp)
        out_str = ""
        while (G_inp[0][0].item() != vocab.stoi[parameters.end_token]) and (G_inp[0][0].item() != vocab.stoi[parameters.pad_token]):
            with T.autograd.no_grad():
                logit, G_hidden, _ = vae(None, G_inp, z, G_hidden)
            probs = F.softmax(logit[0], dim=1)
            G_inp = T.multinomial(probs, 1)
            out_str += (vocab.itos[G_inp[0][0].item()]+" ")
        print(out_str[:-6])
        out.append(out_str[:-6])
    if save:
        original = []
        with open('./data/' + candidates_path, 'r') as fin:
            for line in fin:
                original.append(line.strip())
        fname = './data/' + parameters.dataset + '_candidates.txt'
        with open(fname, 'w') as fout:
            for i in out + original:
                fout.write(i)
                fout.write('\n')


if __name__ == '__main__':
    if parameters.training or parameters.resume_training:
        training()
        generate_sentences(parameters.out_num, save=1)
    else:
        generate_sentences(parameters.out_num, save=1)