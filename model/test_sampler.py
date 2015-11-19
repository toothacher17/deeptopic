from sampler import *
from utils import *

# to test the collapsed gibbs sampler
# use simple save model to save top k words and test the sampler

######### first load word data via utils
word_filename = "../preprocess/data/word_feature2"
word_dictname = "../preprocess/data/word_dict"
word_feature = load_word_data(word_filename)
word_dict = load_dict(word_dictname)


######### initilize a sampler
K = 20                          # topic number
M = len(word_feature)           # file number
V = get_word_size(word_feature) # word size
top_num = 10                    # top number words
beta = 0.1                      # prior beta
iter_num = 200                 # iter num

# initialize the alpha, alpha is the M * K
alpha = []                      # prior alpha
for m in range(M):
    temp = []
    for k in range(K):
        temp.append(2.5)
    alpha.append(temp)

# set sampler
sampler = Sampler(word_feature, K, V, beta)
sampler.init_params()


######### iteration
for iter_num in range(iter_num):
    # star iteration, pass alpha, iter num, and save model flag
    sampler.assigning(alpha, iter_num, 0)


######## save results
sampler.simple_save_model(top_num, word_dict, "lda_top1")
sampler.simple_save_perplexity("lda_stat1")
