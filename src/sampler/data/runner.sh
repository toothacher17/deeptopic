#!/bin/bash

g++ -O3 -std=c++11 preprocessing.cpp -o "fp"

#Download NIPS from UCI repository
wget https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nips.txt.gz
gunzip docword.nips.txt.gz
wget https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.nips.txt
./fp docword.nips.txt vocab.nips.txt nips 500 0
rm -rf docword.nips.txt vocab.nips.txt

wget
https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.enron.txt.gz
gunzip docword.enron.txt.gz
wget
https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.enron.txt
./fp docword.enron.txt vocab.enron.txt enron 500 0
rm -rf docword.enron.txt.gz docword.enron.txt vocab.enron.txt
