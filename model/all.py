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
word2vec_filename = "../preprocess/data/filtered_word2vec"
vec_size = 40
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

# load word2vec
word2vec = load_word2vec(word2vec_filename, V, vec_size)


# debug the size
#print(len(meta_train))
#print(len(meta_test))



########## Init sampler
sampler = Sampler(word_train, word_test, K, V)
sampler.init_params()

# debug the size
#print(sampler.M)
#print(sampler.M_test)
#print(len(word2vec))
#print(len(word2vec[0]))


########## configure NN symbolic
##### init up condition
# add meta data as input
meta_train_input = mx.nd.array(meta_train)
meta_test_input  = mx.nd.array(meta_test)

# init other weight
up_w1_input = init_random_nd(2*K, meta_size, 0.07, 1)
up_w2_input = init_random_nd(K, 2*K, 0.07, 1)
up_b1_input = init_random_nd(2*K, 1, 0.03, 0)
up_b2_input = init_random_nd(K, 1, 0.03, 0)

up_w1_grads = init_zero_nd(2*K, meta_size, 1)
up_w2_grads = init_zero_nd(K, 2*K, 1)
up_b1_grads = init_zero_nd(2*K, 1, 0)
up_b2_grads = init_zero_nd(K, 1, 0)

train_up = Neural(K, train_doc_size, meta_size, meta_train_input, \
                  up_w1_input, up_b1_input, up_w2_input, up_b2_input, \
                  up_w1_grads, up_b1_grads, up_w2_grads, up_b2_grads)

test_up = Neural(K, test_doc_size, meta_size, meta_test_input, \
                 up_w1_input, up_b1_input, up_w2_input, up_b2_input, \
                 up_w1_grads, up_b1_grads, up_w2_grads, up_b2_grads)

#### init down condition
word2vec_input = mx.nd.array(word2vec)

down_w1_input = init_random_nd(2*K, vec_size, 0.0007, 1)
down_w2_input = init_random_nd(K, 2*K, 0.0007, 1)
down_b1_input = init_random_nd(2*K, 1, 0.0003, 0)
down_b2_input = init_random_nd(K, 1, 0.0003, 0)

down_w1_grads = init_zero_nd(2*K, vec_size, 1)
down_w2_grads = init_zero_nd(K, 2*K, 1)
down_b1_grads = init_zero_nd(2*K, 1, 0)
down_b2_grads = init_zero_nd(K, 1, 0)

down = Neural(K, V, vec_size, word2vec_input, \
              down_w1_input, down_b1_input, down_w2_input, down_b2_input, \
              down_w1_grads, down_b1_grads, down_w2_grads, down_b2_grads)

# debug for nn configure
#train_up.infer_shape()
#test_up.infer_shape()
#down.infer_shape()


########## EM Framework for updates
# log likelihood gradient, composed by digamma function
up_llh_grad = mx.nd.empty((train_doc_size,K))
up_llh_grad[:] = np.random.uniform(0.0, 0.001, (train_doc_size,K))

down_llh_grad = mx.nd.empty((V,K))
down_llh_grad[:] = np.random.uniform(0.0, 0.001, (V,K))



for it in range(iter_num):
    #### E step
    # texec forward to get output, then sampling
    train_up.texec.forward()
    alpha_train = train_up.texec.outputs[0].asnumpy()
    test_up.texec.forward()
    alpha_test = test_up.texec.outputs[0].asnumpy()
    down.texec.forward()
    beta = down.texec.outputs[0].asnumpy()
    sampler.assigning(alpha_train, alpha_test, beta, it)

    #### M step
    # first get outer gradients
    up_llh_temp = up_llh_grad.asnumpy()
    up_llh_temp = cal_up_llh_grad(sampler, alpha_train, up_llh_temp)
    up_llh_grad = mx.nd.array(up_llh_temp)

    down_llh_temp = down_llh_grad.asnumpy()
    down_llh_temp = cal_down_llh_grad(sampler, beta, down_llh_temp)
    down_llh_grad = mx.nd.array(down_llh_temp)

    # then update all args
    train_up.texec.backward(out_grads=up_llh_grad)
    down.texec.backward(out_grads=down_llh_grad)

    for name in train_up.args_dict:
        if name != "data":
            #temp_step = step / ((it+1)**2)
            temp_step = step
            SGD(train_up.args_dict[name], train_up.grads_dict[name], \
                temp_step)
            SGD(down.args_dict[name], down.grads_dict[name], \
                temp_step)

#word_dict = load_dict("../preprocess/data/filtered_word_dict")
#sampler.simple_save_model(top_words, word_dict, "nn_top1")
#sampler.simple_save_perplexity("nn_stat1")

#"""

