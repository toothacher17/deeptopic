import numpy as np
import random
import os
import math
import time

# Collapsed gibbs sampler
# sample new topics in each E step
# d is the doc index, D is the corpus doc size
# v is the doc word cnt, V is the corpus word cnt
class Sampler(object):


    # dset: Doc Words, size M * v
    # K: Topic size; V: Word size
    # beta: prior
    def __init__(self, dset, dset_test, K, V):
        self.K = K
        #self.beta = beta
        self.p = []         # store sampling probability, size k
        self.test_iter = 5   # test iteration

        # training related data
        self.dset = dset
        self.M = len(self.dset)
        #self.V = self.get_word_size(self.dset)

        self.Z = []         # store topic assign result, size M * V
        self.nw = []        # store word on topic distribution, size V * K
        self.nwsum = []     # store word sum on each topic, size K
        self.nd = []        # store word on doc distribution, size M * K
        self.ndsum = []     # store word sum on each doc, size M

        self.llhw = []      # log likelihood function

        # test related data
        self.dset_test = dset_test
        self.M_test = len(self.dset_test)
        #self.V_test = self.get_word_size(self.dset_test)

        self.Z_test = []
        self.nw_test = []
        self.nwsum_test = []
        self.nd_test = []
        self.ndsum_test = []

        self.llhw_test = []

        # running time for each iteration
        self.time = []

        # stupid setting....
        self.V = V
        self.V_test = V

    # initialize all parameters
    def init_params(self):
        #### init training related
        # init params
        self.p = [0.0 for x in range(self.K)]
        self.nw = [ [0 for y in range(self.K)] for x in range(self.V) ]
        self.nwsum = [0 for x in range(self.K) ]
        self.nd = [ [0 for y in range(self.K)] for x in range(self.M) ]
        self.ndsum = [0 for x in range(self.M)]

        # initialize topic assign
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

        #### init testing related
        # init params
        self.p_test = [0.0 for x in range(self.K)]
        self.nw_test = [ [0 for y in range(self.K)] for x in range(self.V_test) ]
        self.nwsum_test = [0 for x in range(self.K)]
        self.nd_test = [ [0 for y in range(self.K)] for x in range(self.M_test) ]
        self.ndsum_test = [0 for x in range(self.M_test)]

        # init topic assign
        self.Z_test = [ [] for x in range(self.M_test)]
        for doc_id in range(self.M_test):
            doc_len = len(self.dset_test[doc_id])
            self.Z_test[doc_id] = [0 for y in range(doc_len) ]
            self.ndsum_test[doc_id] = doc_len
            for word_id in range(doc_len):
                topic = random.randint(0, self.K - 1)
                self.Z_test[doc_id][word_id] = topic
                wordmap_id = self.dset_test[doc_id][word_id]
                self.nw_test[wordmap_id][topic] += 1
                self.nwsum_test[topic] += 1
                self.nd_test[doc_id][topic] += 1



    # sampling a new topic for each doc_id, word_id
    # the alpha is get from forward result via NN
    # the data structure of alpha is numpy array
    def sampling_train(self, doc_id, word_id, wordmap_id,\
                       alpha_d, beta_w, beta_sum):

        topic = self.Z[doc_id][word_id]

        # collapsed gibbs sampler, first should minus 1 before sampling
        self.nw[wordmap_id][topic] -= 1
        self.nwsum[topic] -= 1
        self.nd[doc_id][topic] -= 1
        self.ndsum[doc_id] -= 1

        # some param
        Kalpha = np.sum(alpha_d)

        # sample a new topic
        # calculate the probability density for each topic
        for k in range(self.K):
            Vbeta = beta_sum[k]
            beta = beta_w[k]
            self.p[k] = (self.nw[wordmap_id][k]+beta)/(self.nwsum[k]+Vbeta) * \
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


    # sampling training dataset
    # basic algorithm was the same
    def sampling_test(self, doc_id, word_id, wordmap_id, \
                      alpha_d, beta_w, beta_sum):
        # first get prev topic
        # the word id is the word pos id in a doc
        # the wordmap_id is the word id in the word dict
        topic = self.Z_test[doc_id][word_id]

        self.nw_test[wordmap_id][topic] -= 1
        self.nwsum_test[topic] -= 1
        self.nd_test[doc_id][topic] -= 1
        self.ndsum_test[doc_id] -= 1

        # alpha_d is the row for the doc d
        Kalpha = np.sum(alpha_d)

        ## samppling a new topic
        # get prob p
        for k in range(self.K):
            beta = beta_w[k]
            Vbeta = beta_sum[k]
            self.p[k] = \
            (self.nw_test[wordmap_id][k]+beta)/(self.nwsum_test[k]+Vbeta) * \
            (self.nd_test[doc_id][k]+alpha_d[k])/(self.ndsum_test[doc_id]+Kalpha)
        # get the P
        for k in range(1, self.K):
            self.p[k] += self.p[k-1]
        # get a new topic
        u = random.uniform(0, self.p[self.K-1])
        for new_topic in range(self.K):
            if self.p[new_topic]>u:
                break

        # update new states and return the new sampled topic
        self.nw_test[wordmap_id][new_topic] += 1
        self.nwsum_test[new_topic] += 1
        self.nd_test[doc_id][new_topic] += 1
        self.ndsum_test[doc_id] += 1
        return new_topic


    # assigning new topic to each word in each doc
    # the alpha is get from forward result via NN
    # the data structure of alpha is numpy array
    def assigning(self, alpha_train, alpha_test, beta, iter_num):
        print("Sampling iteration %d ..." %iter_num)
        start_time = time.time()

        # first, get the sum of beta for each topic
        beta_sum = beta.sum(axis=0)
        # training, assign new topic to each word in each doc
        for doc_id in range(self.M):
            for word_id in range(len(self.dset[doc_id])):
                wordmap_id = self.dset[doc_id][word_id]

                new_topic = self.sampling_train(doc_id,word_id,wordmap_id,\
                                                alpha_train[doc_id],\
                                                beta[wordmap_id], \
                                                beta_sum)

                self.Z[doc_id][word_id] = new_topic

        llhw = self.cal_llhw(alpha_train, beta, beta_sum)
        self.llhw.append(llhw)
        print("the llhw_train of this iteration is %s" %str(llhw))

        # testing, test the test dataset, sample 10 times
        for it in range(self.test_iter):
            for doc_id in range(self.M_test):
                for word_id in range(len(self.dset_test[doc_id])):
                    wordmap_id = self.dset_test[doc_id][word_id]
                    new_topic=self.sampling_test(doc_id,word_id,wordmap_id, \
                                                alpha_test[doc_id], \
                                                beta[wordmap_id], \
                                                beta_sum)
                    self.Z_test[doc_id][word_id] = new_topic

        llhw_test = self.cal_llhw_test(alpha_test, beta, beta_sum)
        self.llhw_test.append(llhw_test)
        print("the llhw_test of this iteration is %s" %str(llhw_test))

        # get running time
        end_time = time.time()
        used_time = end_time - start_time
        #self.time.append(used_time)
        print("Finish iteration, using time %s" %str(used_time))
        print("----------------------------------")


    # cal the llhw to see the perplexity decreasing
    # the alpha should be given
    def cal_llhw(self, alpha, beta_all, beta_sum):
        result = 0.0
        num_tokens = 0
        # loop all the docs
        for m in range(len(self.ndsum)):
            d_sum = 0.0
            num_tokens += self.ndsum[m]
            alpha_m = np.sum(alpha[m])
            nd_m = self.ndsum[m]
            #Vbeta = self.V * self.beta
            #beta = self.beta
            # loop all the words
            for n in range(len(self.dset[m])):
                w_sum = 0.0
                word_idx = self.dset[m][n]
                # loop all topics
                for k in range(self.K):
                    beta = beta_all[word_idx][k]
                    Vbeta = beta_sum[k]
                    nd_mk = self.nd[m][k]
                    nw_nk = self.nw[word_idx][k]
                    nw_k = self.nwsum[k]
                    alpha_mk = alpha[m][k]
                    w_sum += (alpha_mk+nd_mk)*(beta+nw_nk)/(Vbeta+nw_k)

                w_sum = w_sum / (alpha_m+nd_m)
                d_sum += math.log(w_sum)
            result += d_sum
        return result / num_tokens


    # cal the test llhw
    def cal_llhw_test(self, alpha, beta_all, beta_sum):
        result = 0.0
        num_tokens = 0.0
        # loop all doc
        for m in range(self.M_test):
            d_sum = 0.0     # score for a doc
            num_tokens += self.ndsum_test[m]
            alpha_m = np.sum(alpha[m])
            nd_m = self.ndsum_test[m]
            #beta = self.beta
            #Vbeta = self.V_test * beta
            # loop every word
            for n in range(nd_m):
                w_sum = 0.0     # score for a word
                word_idx = self.dset_test[m][n]
                for k in range(self.K):
                    beta = beta_all[word_idx][k]
                    Vbeta = beta_sum[k]
                    nd_mk = self.nd_test[m][k]
                    nw_nk = self.nw_test[word_idx][k]
                    nw_k = self.nwsum_test[k]
                    alpha_mk = alpha[m][k]
                    w_sum += (alpha_mk+nd_mk)*(beta+nw_nk)/(Vbeta+nw_k)

                w_sum = w_sum / (alpha_m+nd_m)
                d_sum += math.log(w_sum)
            result += d_sum
        return result / num_tokens


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

    # save perplexity
    def simple_save_perplexity(self, filename):
        write_file = open(filename, 'w')
        for i in range(len(self.llhw)):
            write_file.write("Iter " + str(i) + " LLHW " + \
                    str(self.llhw[i]) + "\n")
        for i in range(len(self.llhw_test)):
            write_file.write("Iter " + str(i) + " LLHW_test " + \
                              str(self.llhw_test[i]) + "\n")
        write_file.close()


    # help function to get word size
    # get the V for training and testing data
    def get_word_size(self, word_feature):
        word_set =  set()
        for line in word_feature:
            for index in line:
                word_set.add(index)
        return len(word_set)

