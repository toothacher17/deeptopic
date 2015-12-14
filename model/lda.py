from sampler import *
from utils import *
from random import shuffle

# to test the collapsed gibbs sampler
# use simple save model to save top k words and test the sampler

######### first load word data via utils
word_filename = "../preprocess/data/word_feature2"
word_dictname = "../preprocess/data/word_dict"
test_size = 100
word_feature = load_word_data(word_filename)
word_dict = load_dict(word_dictname)
shuffle(word_feature)
data_test  = word_feature[:test_size]
data_train = word_feature[test_size:]


######### initilize a sampler
K = 20                          # topic number
M = len(word_feature)           # file number
top_num = 10                    # top number words
beta = 0.1                      # prior beta
iter_num = 200                 # iter num
V = get_word_size(word_feature)

# initialize the alpha, alpha is the M * K
alpha = []                      # prior alpha
for m in range(M):
    temp = []
    for k in range(K):
        temp.append(1.0)
    alpha.append(temp)
alpha_test  = alpha[:test_size]
alpha_train = alpha[test_size:]

# set sampler
sampler = Sampler(data_train, data_test, K, beta, V)
sampler.init_params()


######### iteration
for iter_num in range(iter_num):
    # star iteration, pass alpha, iter num, and save model flag
    sampler.assigning(alpha_train, alpha_test, iter_num)


######## save results
sampler.simple_save_model(top_num, word_dict, "lda_top1")
sampler.simple_save_perplexity("lda_stat1")
