from Paper import *
from help_funs import *

stop_words = load_stop_words()

# store metadata index
author_dict = dict()
conf_dict = dict()
org_dict = dict()
year_dict = dict()
word_dict = dict()

f = open("../data/r10k_file")

index = -1
title = ""
author_list = ""
org_list = ""
year = -1
conf = ""
abstract = ""
no = -1


# loop for the first time to build all dicts
for line in f:
    if "-----------------" in line:
        p = Paper(index, title, author_list, org_list, year,\
            conf, abstract, no, stop_words)

        for a in p.author_list:
            if a not in author_dict:
                author_dict[a] = len(author_dict)

        for o in p.org_list:
            if o not in org_dict:
                org_dict[o] = len(org_dict)

        if p.conf not in conf_dict:
            conf_dict[p.conf] = len(conf_dict)

        if p.year not in year_dict:
            year_dict[p.year] = len(year_dict)

        for w in p.abstract.split(" "):
            if w not in word_dict:
                if w != "":
                    word_dict[w] = len(word_dict)

        # reset value
        index = -1
        title = ""
        author_list = ""
        org_list = ""
        year = -1
        conf = ""
        abstract = ""
        no = -1
    else:
        content = line.strip().split(" ")
        if len(content) > 1 :
            if content[0] == "#index" :
                index = str(" ".join(content[1:]))
            if content[0] == "#*" :
                title = str(" ".join(content[1:]))
            if content[0] == "#@" :
                author_list = str(" ".join(content[1:]))
            if content[0] == "#o" :
                org_list = str(" ".join(content[1:]))
            if content[0] == "#t" :
                year = str(" ".join(content[1:]))
            if content[0] == "#c" :
                conf = str(" ".join(content[1:]))
            if content[0] == "#!" :
                abstract = str(" ".join(content[1:]))
            if content[0] == "#n" :
                no = int(content[1])
f.close()


# init file writer, stream writing files
len1 = len(author_dict)
len2 = len(org_dict) + len1
len3 = len(conf_dict) + len2

write_index_dict(author_dict, "data/author_dict_10k", 0)
write_index_dict(org_dict, "data/org_dict_10k", len1)
write_index_dict(conf_dict, "data/conf_dict_10k", len2)
write_index_dict(year_dict, "data/year_dict_10k", len3)
write_index_dict(word_dict, "data/word_dict_10k", 0)

meta_feature = open("data/meta_feature_10k", "w")
word_feature1 = open("data/word_feature1_10k", "w")
word_feature2 = open("data/word_feature2_10k", "w")
word_feature3 = open("data/word_feature3_10k", "w")

# loop for the second time to transfer features
f2 = open("../data/r10k_file")
for line in f2:
    if "-----------------" in line:
        p = Paper(index, title, author_list, org_list, year,\
            conf, abstract, no, stop_words)

        # write metadata file
        meta_list = []
        for a in p.author_list:
            meta_list.append(author_dict[a])
        for o in p.org_list:
            meta_list.append(org_dict[o]+len1)
        meta_list.append(conf_dict[p.conf]+len2)
        meta_list.append(year_dict[p.year]+len3)
        meta_feature.write(" ".join([str(x) for x in meta_list]) + "\n")

        # write words file
        # write word pair
        word_pairs = transfer_word_pair(p.abstract)
        word_feature1.write(word_pairs + "\n")
        # write word index
        word_mapids = transfer_wordmapid(p.abstract, word_dict)
        word_feature2.write(word_mapids + "\n")
        # write word
        word_feature3.write(p.abstract + "\n")

        # reset value
        index = -1
        title = ""
        author_list = ""
        org_list = ""
        year = -1
        conf = ""
        abstract = ""
        no = -1
    else:
        content = line.strip().split(" ")
        if len(content) > 1 :
            if content[0] == "#index" :
                index = str(" ".join(content[1:]))
            if content[0] == "#*" :
                title = str(" ".join(content[1:]))
            if content[0] == "#@" :
                author_list = str(" ".join(content[1:]))
            if content[0] == "#o" :
                org_list = str(" ".join(content[1:]))
            if content[0] == "#t" :
                year = str(" ".join(content[1:]))
            if content[0] == "#c" :
                conf = str(" ".join(content[1:]))
            if content[0] == "#!" :
                abstract = str(" ".join(content[1:]))
            if content[0] == "#n" :
                no = int(content[1])

f2.close()
word_feature1.close()
word_feature2.close()
word_feature3.close()
meta_feature.close()


