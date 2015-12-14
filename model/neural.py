import mxnet as mx
import numpy as np

from utils import *
from nn_utils import *
import time
import random


## NN configuration with mxnet
dev = mx.cpu()
batch_size = 100
step = 0.01

# class used for neural networks configuration
class Neural(object):

    # init the nn
    # K: the topic number, used for num_hidden
    def __init__(self, K, data_size, feature_size, data, \
            w1, b1, w2, b2, w1_grad, b1_grad, w2_grad, b2_grad):
        self.K = K
        self.data_size = data_size
        self.feature_size = feature_size

        self.neural = self.configure_nn()
        self.args_dict = dict()
        self.grads_dict = dict()
        self.reqs = []

        self.bind_data(data)
        self.bind_args(w1, b1, w2, b2)
        self.bind_grads(w1_grad, b1_grad, w2_grad, b2_grad)
        self.init_reqs()

        self.texec = self.bind_exec()

    # configure symbolic configuration for nn
    def configure_nn(self):
        # input
        data = mx.symbol.Variable('data')
        # first layer, num_hidden is 2*K in order not to lose information
        w1 = mx.symbol.Variable('w1')
        fc1 = mx.symbol.FullyConnected(data=data,weight=w1,name='fc1',num_hidden=2*self.K)
        act1 = mx.symbol.Activation(data=fc1,name='act1',act_type="tanh")
        # second layer, num_hidden is K, output is alpha, output is positive
        w2 = mx.symbol.Variable('w2')
        fc2 = mx.symbol.FullyConnected(data=act1,weight=w2,name='fc2',num_hidden=self.K)
        act2 = mx.symbol.Activation(data=fc2,name='act2',act_type="sigmoid")

        return act2

    # infer_shape for the configured nn
    def infer_shape(self):
        data_shape = (self.data_size, self.feature_size)
        arg_shape, out_shape, aux_shape = self.neural.infer_shape(data=data_shape)
        arg_names = self.neural.list_arguments()
        print(dict(zip(arg_names, arg_shape)))
        print(out_shape)

    # bind with data input
    def bind_data(self, data):
        data_grad = mx.nd.zeros((self.data_size, self.feature_size))
        self.args_dict['data'] = data
        self.grads_dict['data'] = data_grad

    # bind with args
    def bind_args(self, w1, b1, w2, b2):
        self.args_dict['w1'] = w1
        self.args_dict['fc1_bias'] = b1
        self.args_dict['w2'] = w2
        self.args_dict['fc2_bias'] = b2


    # bind with grads
    def bind_grads(self, w1, b1, w2, b2):
        self.grads_dict['w1'] = w1
        self.grads_dict['fc1_bias'] = b1
        self.grads_dict['w2'] = w2
        self.grads_dict['fc2_bias'] = b2

    # init with req
    def init_reqs(self):
        self.reqs = ["write" for name in self.args_dict]


    # bind with executor
    def bind_exec(self):
        texec = self.neural.bind(ctx=dev, args=self.args_dict, \
                                 args_grad= self.grads_dict, \
                                 grad_req = self.reqs)
        return texec

