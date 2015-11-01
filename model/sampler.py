import numpy as np
import random
import os


# Collapsed gibbs sampler
# sample new topics in each E step
# d is the doc index, D is the corpus doc size
# v is the doc word cnt, V is the corpus word cnt
class Sampler(object):


    # dset: Doc Words, size M * v
    # K: Topic size; V: Word size
    # beta: prior
    def __init__(self, dset, K, V, beta):
        self.dset = dset
        self.K = K
        self.M = len(self.dset)
        self.V = V
        self.beta = beta

        self.p = []         # store sampling probability, size k
        self.Z = []         # store topic assign result, size M * V
        self.nw = []        # store word on topic distribution, size V * K
        self.nwsum = []     # store word sum on each topic, size K
        self.nd = []        # store word on doc distribution, size M * K
        self.ndsum = []     # store word sum on each doc, size M


    # initialize all parameters
    def init_params(self):
        # initialize params
        self.p = [0.0 for x in range(self.K)]
        self.nw = [ [0 for y in range(self.K)] for x in range(self.V) ]
        self.nwsum = [0 for x in range(self.K) ]
        self.nd = [ [0 for y in range(self.K)] for x in range(self.M) ]
        self.ndsum = [0 for x in range(self.M)]

        # initialize
        self.Z = [ [] for x in range(self.M) ]
        for doc_id in range(self.M):
            self.Z[doc_id] = [0 for y in range(len(self.dset[doc_id]))]
            self.ndsum[doc_id] = len(self.dset[doc_id])
            for word_id in range(len(self.dset[doc_id])):
                # random assign topic to word_id in doc_id
                topic = random.randint(0, self.K -1)
                self.Z[doc_id][word_id] = topic
                wordmap_id = self.dset[doc_id][word_id]
                self.nw[wordmap_id][topic] += 1
                self.nwsum[topic] += 1
                self.nd[doc_id][topic] += 1


    # sampling a new topic for each doc_id, word_id
    # the alpha is get from forward result via NN
    # the data structure of alpha is numpy array
    def sampling(self, doc_id, word_id, alpha_d):
        topic = self.Z[doc_id][word_id]
        wordmap_id = self.dset[doc_id][word_id]

        # collapsed gibbs sampler, first should minus 1 before sampling
        self.nw[wordmap_id][topic] -= 1
        self.nwsum[topic] -= 1
        self.nd[doc_id][topic] -= 1
        self.ndsum[doc_id] -= 1

        # some param
        Vbeta = self.V * self.beta
        Kalpha = np.sum(alpha_d)

        # sample a new topic
        # calculate the probability density for each topic
        for k in range(self.K):
            self.p[k] = (self.nw[wordmap_id][k]+self.beta)/(self.nwsum[k]+Vbeta) * \
                        (self.nd[doc_id][k]+alpha_d[k])/(self.ndsum[doc_id]+Kalpha)
        # calculate the probability accum for each topic
        for k in range(1,self.K):
            self.p[k] += self.p[k-1]
        u = random.uniform(0, self.p[self.K-1])
        for new_topic in range(self.K):
            if self.p[new_topic]>u:
                break

        # update new states and return the new sampled topic
        self.nw[wordmap_id][new_topic] += 1
        self.nwsum[new_topic] += 1
        self.nd[doc_id][new_topic] += 1
        self.ndsum[doc_id] += 1
        return new_topic


    # assigning new topic to each word in each doc
    # the alpha is get from forward result via NN
    # the data structure of alpha is numpy array
    def assigning(self, alpha, iter_num, save_model_flag):
        print("Sampling iteration %d ..." %iter_num)

        # assign new topic to each word in each doc
        for doc_id in range(self.M):
            for word_id in range(len(self.dset[doc_id])):
                new_topic = self.sampling(doc_id, word_id, alpha[doc_id])
                self.Z[doc_id][word_id] = new_topic

        #if save_model_flag:
        #    self.save_model()
        print("Finish iteration")


    # simple save model function, save the top 10 words for each topic
    def simple_save_model(self, top_word_num, word_dict, filename):
        write_file = open(filename, 'w')
        # for each topic, write top num words
        for topic in range(self.K):
            write_file.write("Topic " + str(topic) + "top words:\n")
            top_words = []
            for word in range(self.V):
                top_words.append((word, self.nw[word][topic]))
            # sort to get top k words
            top_words.sort(key=lambda x:x[1], reverse=True)
            for i in range(top_word_num):
                word_key = str(word_dict[top_words[i][0]])
                word_val = str(top_words[i][1])
                write_file.write("\t" + word_key + ": " + word_val + "\n")
        write_file.close()
