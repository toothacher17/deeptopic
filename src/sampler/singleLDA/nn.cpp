#include <iostream>

using namespace std;

#include "lda.h"

#include "mxnet/ndarray.h"
#include "mxnet/base.h"


/* This file was used to build the deep topic model */

int main(int argc, char ** argv) {
    /* set up some basic args */
    int K = 20;
    float beta = 0.1;
    int iter_num = 200;
    int top_words = 10;

    /* init the sampler */
    model *lda=NULL;
    if (!(lda = model::init(argc, argv))) {
        return 1;
    }
    int train_doc_size = lda->trngdata->M;
    int test_doc_size = lda->testdata->M;

    // for debug
    cout << "training file size " << train_doc_size << endl;
    cout << "testing file size "  << test_doc_size  << endl;


    /* load meta_data into a NDArray */
    // test NDArray
    const int m = 3;
    const int n = 2;
    mxnet::TShape shape = mshadow::Shape2(m,n);
    mxnet::Context ctx = mxnet::Context::Create(mxnet::Context::kCPU, 1);
    mxnet::NDArray a(shape, ctx, false);
    mxnet::NDArray b(shape, ctx, false);
    mxnet::real_t* aptr = static_cast<mxnet::real_t*>(a.data().dptr_);
    mxnet::real_t* bptr = static_cast<mxnet::real_t*>(b.data().dptr_);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            aptr[i*n + j] = i*n + j;
            bptr[i*n + j] = i*n + 2*j;
        }
    }

    mxnet::NDArray c = b;
    c += a; // this line would cause the compiling fail
    c.WaitToRead();
    mxnet::real_t* cptr = static_cast<mxnet::real_t*>(c.data().dptr_);
    cout << "NDArray value! " <<cptr[m*n - 1] << endl;

    // configure and build neural networks

    // bind with executors

    // EM Framewrok to update
    return 0;
}

