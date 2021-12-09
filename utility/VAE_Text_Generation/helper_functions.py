import torch as T
import tensorflow as tf


def get_cuda(tensor):
    #Use `tf.config.list_physical_devices('GPU')` instead.
    print(tf.config.list_physical_devices('GPU'))
    if tf.test.is_gpu_available(cuda_only=True):
        tensor = tensor
    return tensor

def get_sentences_in_batch(x, vocab):
    for sent in x:
        str1 = ""
        for word in sent:
            str1 += vocab.itos[word] + " "
        print(str1)
