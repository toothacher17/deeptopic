import numpy as np
import mxnet as mx

# load metadata as features, input to the NN
# the input should use mxnet data structure
def load_data(filename, doc_size, feature_size):
    result = np.zeros((doc_size, feature_size), dtype=np.int)
    meta_file = open(filename)

    index = 0
    for line in meta_file:
        indices = [int(x) for x in line.strip().split(" ")]
        for i in indices:
            result[index][i] = 1
        index += 1

    meta_file.close()
    return result

