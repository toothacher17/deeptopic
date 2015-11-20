import mxnet as mx
import numpy as np

from sampler import *
from utils import *
from nn_utils import *
import time


########### model parameters and configurations
# sampler related args
K = 20
beta = 0.1
iter_num = 200
top_words = 10

# load small data
meta_filename = "../preprocess/data/meta_feature"
word_filename = "../preprocess/data/word_feature2"

# load bigger data
#meta_filename = "../preprocess/meta_feature_50k"
#word_filename = "../preprocess/word_feature2_50k"

word_feature = load_word_data(word_filename)
doc_size = len(word_feature)
meta_size = get_meta_size(meta_filename)
V = get_word_size(word_feature) # V is the word size


########## Init sampler
sampler = Sampler(word_feature, K, V, beta)
sampler.init_params()


########## configure NN symbolic
## NN configuration with mxnet
dev = mx.cpu()
batch_size = 100
step = 0.01

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

## bind the layers with exexcutor
# the arg includes meta, w1, fc1_bias, w2, fc2_bias
meta_shape = (doc_size, meta_size)
arg_shape, out_shape, aux_shape = act2.infer_shape(meta=meta_shape)
arg_names = act2.list_arguments()
print(dict(zip(arg_names, arg_shape)))
print(out_shape)

# add meta data as input
meta_matrix = load_meta_data(meta_filename, doc_size, meta_size)
meta_input = mx.nd.array(meta_matrix)
meta_grad = mx.nd.zeros((doc_size,meta_size))

# init other weight
w1_input = mx.nd.empty((2*K,meta_size))
w1_input[:] = np.random.uniform(-0.07, 0.07, (2*K,meta_size))
w1_grad = mx.nd.zeros((2*K,meta_size))
w2_input = mx.nd.empty(((K,2*K)))
w2_input[:] = np.random.uniform(-0.07, 0.07, (K,2*K))
w2_grad = mx.nd.zeros((K,2*K))
b1_input = mx.nd.empty(((2*K,)))
b1_input[:] = np.random.uniform(-0.03, 0.03, (2*K,))
b1_grad = mx.nd.zeros((2*K,))
b2_input = mx.nd.empty(((K,)))
b2_input[:] = np.random.uniform(-0.03, 0.03, (K,))
b2_grad = mx.nd.zeros((K,))

# bind with executor
args = dict()
args['meta'] = meta_input
args['w1'] = w1_input
args['w2'] = w2_input
args['fc1_bias'] = b1_input
args['fc2_bias'] = b2_input

grads = dict()
grads['meta'] = meta_grad
grads['w1'] = w1_grad
grads['w2'] = w2_grad
grads['fc1_bias'] = b1_grad
grads['fc2_bias'] = b2_grad

reqs = ["write" for name in grads]
texec = act2.bind(ctx=dev, args=args, args_grad=grads, grad_req=reqs)


########## EM Framework for updates
# log likelihood gradient, composed by digamma function
llh_grad = mx.nd.empty((doc_size,K))
llh_grad[:] = np.random.uniform(0.0, 0.001, (doc_size,K))

for it in range(iter_num):
    #### E step
    # texec forward to get output, then sampling
    texec.forward()
    alpha = texec.outputs[0].asnumpy()
    sampler.assigning(alpha, it)

    #### M step
    # first get outer gradients
    llh_temp = llh_grad.asnumpy()
    llh_temp = cal_llh_grad(sampler, alpha, llh_temp)
    llh_grad = mx.nd.array(llh_temp)
    # then update all args
    texec.backward(out_grads=llh_grad)
    for name in arg_names:
        if name != "meta":
            #temp_step = step / ((it+1)**2)
            temp_step = step
            SGD(args[name], grads[name], temp_step)

word_dict = load_dict("../preprocess/data/word_dict")
sampler.simple_save_model(top_words, word_dict, "nn_top1")
sampler.simple_save_perplexity("nn_stat1")



