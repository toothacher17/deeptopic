import mxnet as mx
import numpy as np

from data import mnist_iterator
from sampler import *
from utils import *

########### model parameters and configurations
# sampler related args
K = 50
beta = 0.1
iter_num = 20
top_words = 10

# file related args
meta_filename = "../preprocess/meta_feature"
word_filename = "../preprocess/word_feature2"


doc_size = 1000
meta_size = 3773


########## configure NN symbolic
# NN configuration with mxnet
dev = mx.cpu()
batch_size = 100

# input
meta = mx.symbol.Variable('meta')
# first layer, num_hidden is 2*K in order not to lose information
w1 = mx.symbol.Variable('w1')
fc1 = mx.symbol.FullyConnected(data=meta,weight=w1,name='fc1',num_hidden=2*K)
act1 = mx.symbol.Activation(data=fc1,name='act1',act_type="tanh")
# second layer, num_hidden is K, output is alpha, output is positive
w2 = mx.symbol.Variable('w2')
fc2 = mx.symbol.FullyConnected(data=act1,weight=w2,name='fc2',num_hidden=K)
act2 = mx.symbol.Activation(data=fc2,name='act2',act_type="sigmoid")
## using softmax to avoid 'explosion'
#output = mx.symbol.Softmax(data=act2,name='alphann')

# bind the layers with exexcutor
data_shape =
arg_names = act2.list_arguments()




########## EM Framework for updates





