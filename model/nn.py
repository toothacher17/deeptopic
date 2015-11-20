import mxnet as mx
import numpy as np

from sampler import *
from utils import *
from nn_utils import *
import time
import random


########### model parameters and configurations
# sampler related args
K = 20
beta = 0.1
iter_num = 200
top_words = 10

# load small data
meta_filename = "../preprocess/data/meta_feature"
word_filename = "../preprocess/data/word_feature2"
test_size = 100

# load bigger data
#meta_filename = "../preprocess/meta_feature_50k"
#word_filename = "../preprocess/word_feature2_50k"
#test_size = 500

# feature space size
word_feature = load_word_data(word_filename)
V = get_word_size(word_feature)          # V is total word size
meta_size = get_meta_size(meta_filename) # meta size is total meta size

# load word feature, meta feature, and split into train and test
word_train = []
word_test  = []
shuffle_idx_list = [x for x in range(len(word_feature))]
random.shuffle(shuffle_idx_list)
for i in range(test_size):
    shuffle_idx = shuffle_idx_list[i]
    word_test.append(word_feature[shuffle_idx])
for j in range(test_size, len(word_feature)):
    shuffle_idx = shuffle_idx_list[j]
    word_train.append(word_feature[shuffle_idx])
train_doc_size = len(word_train)
test_doc_size = len(word_test)

# load meta data and split
meta_matrix = load_meta_data(meta_filename, len(word_feature), meta_size)
meta_train = np.zeros((train_doc_size, meta_size), dtype=np.int)
meta_test = np.zeros((test_doc_size, meta_size), dtype=np.int)
for i in range(test_size):
    shuffle_idx = shuffle_idx_list[i]
    meta_test[i] = meta_matrix[shuffle_idx]
for j in range(test_size, len(word_feature)):
    shuffle_idx = shuffle_idx_list[j]
    meta_train[i-test_size] = meta_matrix[shuffle_idx]

# debug the size
#print(len(meta_train))
#print(len(meta_test))



########## Init sampler
sampler = Sampler(word_train, word_test, K, beta, V)
sampler.init_params()

# debug the size
#print(sampler.M)
#print(sampler.M_test)



########## configure NN symbolic
## NN configuration with mxnet
dev = mx.cpu()
batch_size = 100
step = 0.01

## configure networks
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

## bind the layers with exexcutor
# the arg includes meta, w1, fc1_bias, w2, fc2_bias
train_shape = (train_doc_size, meta_size)
test_shape = (test_doc_size, meta_size)

# debug get the size
# get the train size
arg_shape, out_shape, aux_shape = act2.infer_shape(meta=train_shape)
arg_names = act2.list_arguments()
print(dict(zip(arg_names, arg_shape)))
print(out_shape)
# get the test size
arg_shape, out_shape, aux_shape = act2.infer_shape(meta=test_shape)
arg_names = act2.list_arguments()
print(dict(zip(arg_names, arg_shape)))
print(out_shape)


# add meta data as input
meta_train_input = mx.nd.array(meta_train)
meta_test_input  = mx.nd.array(meta_test)
meta_train_grad = mx.nd.zeros((train_doc_size,meta_size))
meta_test_grad  = mx.nd.zeros((test_doc_size,meta_size))

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

## bind with executor
# first bind with train executor
train_args = dict()
train_args['meta'] = meta_train_input
train_args['w1'] = w1_input
train_args['w2'] = w2_input
train_args['fc1_bias'] = b1_input
train_args['fc2_bias'] = b2_input

train_grads = dict()
train_grads['meta'] = meta_train_grad
train_grads['w1'] = w1_grad
train_grads['w2'] = w2_grad
train_grads['fc1_bias'] = b1_grad
train_grads['fc2_bias'] = b2_grad

train_reqs = ["write" for name in train_grads]
train_texec = act2.bind(ctx=dev, args=train_args, \
                        args_grad=train_grads, grad_req=train_reqs)

# then bind with test executor
test_args = dict()
test_args['meta'] = meta_test_input
test_args['w1'] = w1_input
test_args['w2'] = w2_input
test_args['fc1_bias'] = b1_input
test_args['fc2_bias'] = b2_input

test_grads = dict()
test_grads['meta'] = meta_test_grad
test_grads['w1'] = w1_grad
test_grads['w2'] = w2_grad
test_grads['fc1_bias'] = b1_grad
test_grads['fc2_bias'] = b2_grad

test_reqs = ["write" for name in test_grads]
test_texec = act2.bind(ctx=dev, args=test_args, \
                        args_grad=test_grads, grad_req=test_reqs)


########## EM Framework for updates
# log likelihood gradient, composed by digamma function
llh_grad = mx.nd.empty((train_doc_size,K))
llh_grad[:] = np.random.uniform(0.0, 0.001, (train_doc_size,K))

for it in range(iter_num):
    #### E step
    # texec forward to get output, then sampling
    train_texec.forward()
    alpha_train = train_texec.outputs[0].asnumpy()
    test_texec.forward()
    alpha_test = test_texec.outputs[0].asnumpy()
    sampler.assigning(alpha_train, alpha_test,it)

    #### M step
    # first get outer gradients
    llh_temp = llh_grad.asnumpy()
    llh_temp = cal_llh_grad(sampler, alpha_train, llh_temp)
    llh_grad = mx.nd.array(llh_temp)
    # then update all args
    train_texec.backward(out_grads=llh_grad)
    for name in arg_names:
        if name != "meta":
            #temp_step = step / ((it+1)**2)
            temp_step = step
            SGD(train_args[name], train_grads[name], temp_step)

word_dict = load_dict("../preprocess/data/word_dict")
sampler.simple_save_model(top_words, word_dict, "nn_top1")
sampler.simple_save_perplexity("nn_stat1")

#"""

