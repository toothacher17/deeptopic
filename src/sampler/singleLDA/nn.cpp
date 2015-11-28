#include "lda.h"

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
    int train_doc_size = lda->trngdata->M;
    int test_doc_size = lda->testdata->M;

    // for debug
    std::cout << train_doc_size << std::endl;
    std::cout << test_doc_size << std::endl;


    return 0;
}

