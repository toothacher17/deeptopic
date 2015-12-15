import mxnet as mx
import numpy as np

from sampler import *
from neural import *
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
word_filename = "../preprocess/data/filtered_word_feature2"
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


# initialize the beta, beta is V * K
beta = np.zeros((V,K),dtype=np.float)
for v in range(V):
    for k in range(K):
        beta[v][k] = 0.1


########## Init sampler
sampler = Sampler(word_train, word_test, K, V)
sampler.init_params()

# debug the size
#print(sampler.M)
#print(sampler.M_test)


########## configure NN symbolic
# add meta data as input
meta_train_input = mx.nd.array(meta_train)
meta_test_input  = mx.nd.array(meta_test)

# init other weight
w1_input = init_random_nd(2*K, meta_size, 0.07, 1)
w2_input = init_random_nd(K, 2*K, 0.07, 1)
b1_input = init_random_nd(2*K, 1, 0.03, 0)
b2_input = init_random_nd(K, 1, 0.03, 0)

w1_grads = init_zero_nd(2*K, meta_size, 1)
w2_grads = init_zero_nd(K, 2*K, 1)
b1_grads = init_zero_nd(2*K, 1, 0)
b2_grads = init_zero_nd(K, 1, 0)

train_up = Neural(K, train_doc_size, meta_size, meta_train_input, \
                  w1_input, b1_input, w2_input, b2_input, \
                  w1_grads, b1_grads, w2_grads, b2_grads)

test_up = Neural(K, test_doc_size, meta_size, meta_test_input, \
                  w1_input, b1_input, w2_input, b2_input, \
                  w1_grads, b1_grads, w2_grads, b2_grads)


# debug nn config
#train_up.infer_shape()
#test_up.infer_shape()


########## EM Framework for updates
# log likelihood gradient, composed by digamma function
llh_grad = mx.nd.empty((train_doc_size,K))
llh_grad[:] = np.random.uniform(0.0, 0.001, (train_doc_size,K))

for it in range(iter_num):
    #### E step
    # texec forward to get output, then sampling
    train_up.texec.forward()
    alpha_train = train_up.texec.outputs[0].asnumpy()
    test_up.texec.forward()
    alpha_test = test_up.texec.outputs[0].asnumpy()
    sampler.assigning(alpha_train, alpha_test, beta, it)

    #### M step
    # first get outer gradients
    llh_temp = llh_grad.asnumpy()
    llh_temp = cal_up_llh_grad(sampler, alpha_train, llh_temp)
    llh_grad = mx.nd.array(llh_temp)
    # then update all args
    train_up.texec.backward(out_grads=llh_grad)
    for name in train_up.args_dict:
        if name != "data":
            #temp_step = step / ((it+1)**2)
            temp_step = step
            SGD(train_up.args_dict[name], train_up.grads_dict[name], \
                temp_step)

word_dict = load_dict("../preprocess/data/filtered_word_dict")
sampler.simple_save_model(top_words, word_dict, "up_top1")
sampler.simple_save_perplexity("up_stat1")

#"""

