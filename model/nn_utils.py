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
