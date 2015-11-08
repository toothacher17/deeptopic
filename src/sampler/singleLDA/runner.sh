#!/bin/bash

#DATASETS="nips enron"
DATASETS="enron"
METHODS="FTreeLDA"
NUM_ITER="200"
NUM_TOPICS="50"

for DATASET in $DATASETS
do
for METHOD in $METHODS
do
	#Create the directory structure
	time_stamp=`date "+%b_%d_%Y_%H.%M.%S"`
	DIR_NAME='res'/$DATASET/$METHOD/$time_stamp/
	mkdir -p $DIR_NAME

	#save details about experiments in an about file
	echo 'with results being stored in' $DIR_NAME

	#run
	./LDA --method "$METHOD" --testing-mode net --num-topics $NUM_TOPICS --num-iterations $NUM_ITER --output-model $DIR_NAME --num-top-words 15 --training-file ../test_data/"$DATASET".train --testing-file ../test_data/"$DATASET".test | tee -a $DIR_NAME/log.txt
done
done

echo 'done'
