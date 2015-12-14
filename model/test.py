import mxnet as mx
import numpy as np

from utils import *



result = load_word2vec("../preprocess/data/filtered_word2vec", \
                       8550, 40)
print(str(result[0][2]))
print(str(result[2][2]))
print(len(result.sum(axis=0)))

