import re

class Paper:
    index = -1
    title = ""
    author_list = []
    org_list = []
    year = -1
    conf = ""
    abstract = ""
    no = -1
    word_size = 0

    def __init__(self, index, title, author_list, org_list, year, conf, \
            abstract, no, stop_words) :
        self.index = int(index)
        self.title = title
        self.author_list = self.transfer_list(author_list, "author")
        self.org_list = self.transfer_list(org_list)
        self.year = int(year)
        self.conf = conf
        self.abstract = self.filter_abstract(abstract, stop_words)
        self.no = no
        self.word_size = self.get_word_size()

    # transfer author string to author list
    # ignore name disambiguity problem, splited by ';'
    # first char of first name + last name
    # str_type = author or org
    def transfer_list(self, author_list, str_type="org") :
        result_list = []
        str_list = author_list.strip().split(";")
        for origin_str in str_list:
            if str_type == "author" :
                name_parts = origin_str.split(" ")
                if len(name_parts) > 1:
                    first_name = name_parts[0]
                    last_name = name_parts[-1]
                    if len(first_name) > 0 :
                        result_list.append(first_name[0] + last_name)
                    else:
                        result_list.append(last_name)
                else:
                    result_list.append(origin_str)
            else:
                result_list.append(origin_str)
        return result_list


    # keep only english words
    # filter stop words, do I need to filter words occur only one time?
    def filter_abstract(self, str_abstract, stop_words):
        result_list = []
        str_english = "".join(i if (ord(i) < 123 and ord(i) > 96) or \
            (ord(i) < 91 and ord(i) > 64) else " " for i in str_abstract)
        word_list = str_english.strip().split(' ')
        for w in word_list:
            w = w.lower()
            if w not in stop_words and w != " ":
                result_list.append(w)
        return " ".join(r for r in result_list)


    def print_value(self):
        print("#index " + str(self.index))
        print("#* " + str(self.title))
        print("#@ " + str(";".join(x for x in self.author_list)))
        print("#o " + str(";".join(x for x in self.org_list)))
        print("#t " + str(self.year))
        print("#c " + str(self.conf))
        print("#! " + str(self.abstract))
        print("#n " + str(self.no))
        print("--------------------------------")


    def get_word_size(self):
        return len(self.abstract.split(" "))


