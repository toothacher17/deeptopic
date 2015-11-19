#include "lda.h"
#include "digamma.h"

// This file was used to build the deep topic model

int main(int argc, char ** argv) {
    // set up some basic args
    int K = 20;
    float beta = 0.1;
    int iter_num = 200;
    int top_words = 10;

    // init the sampler
    model *lda=NULL;
    if (!(lda = model::init(argc, argv))) {
        return 1;
    }
    int doc_size = lda->trngdata->M;

    std::cout << doc_size << std::endl;

    // first read the metadata, implemented in tutils
    // string filename = "";
    // int ** meta = read_meta();





}

