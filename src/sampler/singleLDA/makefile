CDIR = ../commons

GNU_CC = g++
GNU_CFLAGS = -I"$(CDIR)" -O3 -march=core-avx2 -pthread -std=c++11 -c
GNU_LFLAGS = -I"$(CDIR)" -O3 -march=core-avx2 -pthread -std=c++11

gcc: CC=$(GNU_CC)
gcc: CFLAGS=$(GNU_CFLAGS)
gcc: LFLAGS=$(GNU_LFLAGS)
gcc: LDA
gcc: NN


NN : stringtokenizer.o dataset.o fTree.o utils.o model.o lda.o nn.o
	$(CC) $(LFLAGS) nn.o lda.o model.o fTree.o utils.o stringtokenizer.o dataset.o -o "NN"

LDA : stringtokenizer.o dataset.o fTree.o utils.o model.o lda.o source.o
	$(CC) $(LFLAGS) source.o lda.o model.o fTree.o utils.o stringtokenizer.o dataset.o -o "LDA"

nn.o : nn.cpp lda.h
	$(CC) $(CFLAGS) nn.cpp

source.o : source.cpp lda.h
	$(CC) $(CFLAGS) source.cpp

lda.o : lda.h lda.cpp model.h $(CDIR)/fTree.h $(CDIR)/utils.h
	$(CC) $(CFLAGS) lda.cpp

model.o : model.h model.cpp $(CDIR)/dataset.h $(CDIR)/utils.h
	$(CC) $(CFLAGS) model.cpp

fTree.o : $(CDIR)/fTree.h $(CDIR)/fTree.cpp $(CDIR)/utils.h
	$(CC) $(CFLAGS) $(CDIR)/fTree.cpp

dataset.o : $(CDIR)/dataset.h $(CDIR)/dataset.cpp $(CDIR)/stringtokenizer.h
	$(CC) $(CFLAGS) $(CDIR)/dataset.cpp

utils.o : $(CDIR)/utils.h $(CDIR)/utils.cpp $(CDIR)/stringtokenizer.h
	$(CC) $(CFLAGS) $(CDIR)/utils.cpp

stringtokenizer.o : $(CDIR)/stringtokenizer.h $(CDIR)/stringtokenizer.cpp
	$(CC) $(CFLAGS) $(CDIR)/stringtokenizer.cpp

clean:
	rm -rf *.o LDA *.obj LDA.exe NN
