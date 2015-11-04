OBJS = main.o data.o
CC = g++
CFLAGS = -O3 -std=c++0x -pg -Wall -c

main : main.o evaluation.o model.o data.o
	$(CC) main.o evaluation.o model.o data.o -o main

main.o : main.cc evaluation.o model.o data.o
	$(CC) $(CFLAGS) main.cc

evaluation.o : model.h evaluation.h evaluation.cc
	$(CC) $(CFLAGS) evaluation.cc

model.o : data.h model.h model.cc
	$(CC) $(CFLAGS) model.cc

data.o : data.h data.cc
	$(CC) $(CFLAGS) data.cc




clean:
	\rm *.o main
