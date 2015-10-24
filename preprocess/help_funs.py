from Paper import *

# load stop words into memory
def load_stop_words():
    stop_words_file = open('../data/stop_words', "r")
    stop_words = set(x.strip() for x in \
            stop_words_file.read().strip().split('\n'))
    stop_words_file.close()
    return stop_words


# write out index map function
def write_index_dict(data, filename, index_base):
    f = open(filename, "w")
    for k,v in data.items():
        f.write(str(k) + "\t" + str(v+index_base) + "\n")
    f.close()

# transfer word line to word pairs
def transfer_word_pair(line):
    result_list = []
    result_dict = dict()
    for w in line.split(" "):
        if w not in result_dict:
            result_dict[w] = 1
        else:
            result_dict[w] += 1
    for k,v in result_dict.items():
        if k != "":
            result_list.append(str(k) + ":" + str(v))
    return " ".join(result_list)

def transfer_wordmapid(line, wordmap):
    result_list = []
    for w in line.split(" "):
        if w != "":
            result_list.append(str(wordmap[w]))
    return " ".join(result_list)


# load paper data into memory
def load_data(filename):
    result = []

    stop_words = load_stop_words

    f = open(filename)

    index = -1
    title = ""
    author_list = ""
    org_list = ""
    year = -1
    conf = ""
    abstract = ""
    no = -1

    for line in f:
        if "-----------------" in line:
            p = Paper(index, title, author_list, org_list, year,\
                conf, abstract, no, stop_words)
            result.append(p)
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
    return result



