import numpy as np
import mxnet as mx

# load metadata as features, input to the NN
# the input should use mxnet data structure
def load_meta_data(filename, doc_size, feature_size):
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


# load word data for collapsed gibbs sampler
# store the data as sparse vector
def load_word_data(filename):
    result = []
    word_file = open(filename)

    for line in word_file:
        result.append([int(x) for x in line.strip().split(" ")])

    word_file.close()
    return result


# load meta word feature dict
def load_dict(filename):
    result = dict()
    f = open(filename)
    for l in f:
        temp =  l.strip().split("\t")
        result[int(temp[1])] = temp[0]
    f.close()
    return result

# get word set size, input is a spare matrix
def get_word_size(word_feature):
    word_set =  set()
    for line in word_feature:
        for index in line:
            word_set.add(index)
    return len(word_set)

def get_meta_size(filename):
    f = open(filename)
    result = set()
    for l in f:
        for x in l.strip().split(" "):
            result.add(x)
    f.close()
    return len(result)
