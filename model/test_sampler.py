from sampler import *
from utils import *

# to test the collapsed gibbs sampler
# use simple save model to save top k words and test the sampler

######### first load word data via utils
word_feature = load_word_data("../preprocess/word_feature2")
word_dict = load_dict("../preprocess/word_dict")


######### initilize a sampler
K = 10                          # topic number
M = len(word_feature)           # file number
top_num = 20                    # top number words
beta = 0.1                      # prior beta

# calculate the V
word_set = set()
for line in word_feature:
    for index in line:
        word_set.add(index)
V = len(word_set)               # word size

# initialize the alpha, alpha is the M * K
alpha = []                      # prior alpha
for m in range(M):
    temp = []
    for k in range(K):
        temp.append(5)
    alpha.append(temp)


# set sampler
sampler = Sampler(word_feature, K, V, beta)
sampler.init_params()


######### iteration
for iter_num in range(100):
    # star iteration, pass alpha, iter num, and save model flag
    sampler.assigning(alpha, iter_num, 0)

sampler.simple_save_model(top_num, word_dict, "10_iter")
