import mxnet as mx
import numpy as np
from digamma import digamma

# perform sgd to update weights
def SGD(weight, grad, step):
    weight[:] -= step * grad

# get llh grad for up neural nets
def cal_up_llh_grad(sampler, alpha, grad):
    for m in range(len(alpha)):
        alpha_m = np.sum(alpha[m])
        wc_m = sampler.ndsum[m]
        for k in range(len(alpha[m])):
            alpha_mk = alpha[m][k]
            wc_mk = sampler.nd[m][k]
            grad[m][k] = digamma(alpha_m) - digamma(wc_m+alpha_m) + \
                         digamma(alpha_mk+wc_mk) - digamma(alpha_mk)
    return grad


# get llh grad for down neural nets
def cal_down_llh_grad(sampler, beta, grad):
    beta_sum = beta.sum(axis=0)
    for v in range(len(beta)):
        for k in range(sampler.K):
            beta_k = beta_sum[k]
            beta_vk = beta[v][k]
            wc_vk = sampler.nw[v][k]
            wc_k = sampler.nwsum[k]
            grad[v][k] = digamma(beta_k) - digamma(beta_k+wc_k) + \
                         digamma(beta_vk+wc_vk) - digamma(beta_vk)

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





