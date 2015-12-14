import mxnet as mx
import numpy as np
from digamma import digamma

# perform sgd to update weights
def SGD(weight, grad, step):
    weight[:] -= step * grad

# get llh grad for neural nets
def cal_llh_grad(sampler, alpha, grad):
    for i in range(len(alpha)):
        alpha_i = np.sum(alpha[i])
        wc_i = sampler.ndsum[i]
        for j in range(len(alpha[i])):
            alpha_ij = alpha[i][j]
            wc_ij = sampler.nd[i][j]
            grad[i][j] = digamma(alpha_i) - digamma(wc_i+alpha_i) + \
                          digamma(alpha_ij+wc_ij) - digamma(alpha_ij)
    return grad

# init mxnet random ndarray
# init_type = 0, (col_size,)
# init_type = 1, (col_size, row_size)
def init_random_nd(col_size, row_size, rrange, init_type):
    if init_type == 1:
        data = mx.nd.empty((col_size, row_size))
        data[:] = np.random.uniform(0 - rrange, rrange, (col_size, row_size))
    else:
        data = mx.nd.empty((col_size,))
        data[:] = np.random.uniform(0 - rrange, rrange, (col_size,))

    return data


# init mxnet zero ndarray
def init_zero_nd(col_size, row_size, init_type):
    if init_type == 1:
        return mx.nd.zeros((col_size, row_size))
    else:
        return mx.nd.zeros((col_size,))





