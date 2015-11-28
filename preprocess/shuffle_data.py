# use this script to shuffle data, and split into train and test data
import random

## config
# large dataset
file_size = 10000       # total file_size
test_size = 500         # test data_size
word_file = "./data/word_feature3_10k"
meta_file = "./data/meta_feature_10k"
word_train = "./data/shuffle/word_10k_train"
word_test = ".//data/shuffle/word_10k_test"
meta_train = "./data/shuffle/meta_10k_train"
meta_test = "./data/shuffle/meta_10k_test"

# small dataset

# first build random list
random_idx_set = set()
while len(random_idx_set) < test_size:
    random_idx_set.add(random.randint(0, file_size))



## streaming read data based on random index
# add word file
f = open(word_file)
w1 = open(word_train, 'w')
w2 = open(word_test, 'w')
idx = 0
for l in f:
    if idx not in random_idx_set:
        w1.write(l.strip() + '\n')  # write to the train file
    else:
        w2.write(l.strip() + '\n')  # write to the test file
    idx += 1

f.close()
w1.close()
w2.close()

# add meta file
f = open(meta_file)
w1 = open(meta_train, 'w')
w2 = open(meta_test, 'w')
idx = 0
for l in f:
    if idx not in random_idx_set:
        w1.write(l.strip() + '\n')  # write to the train file
    else:
        w2.write(l.strip() + '\n')  # write to the test file
    idx += 1

f.close()
w1.close()
w2.close()
