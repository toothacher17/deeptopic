#include <iostream>

using namespace std;

#include "lda.h"

#include "mxnet/ndarray.h"
#include "mxnet/base.h"
#include "mxnet/operator.h"
#include "mxnet/symbolic.h"

/*Define basic params*/
const int K = 50;           // topic number
const int iter_num = 200;   // iterations
const int top_words = 10;   // top save words

/* Basic NN class for configuration*/
class NN {
    public:
        /* neural networks layers */
        // FullyConnected layer
        mxnet::Symbol FullyConnectedLayer(mxnet::Symbol input,
                string num_hidden="28", string name="fc") {

            mxnet::OperatorProperty *fully_connected_op =
                mxnet::OperatorProperty::Create("FullyConnected");

            vector<pair<string, string>> fc_config;
            fc_config.push_back(make_pair("num_hidden", num_hidden));
            fc_config.push_back(make_pair("no_bias", "false"));
            fully_connected_op->Init(fc_config);

            vector<mxnet::Symbol> sym_vec;
            sym_vec.push_back(input);

            mxnet::Symbol fc = mxnet::Symbol::Create(fully_connected_op)(sym_vec,name);
            return fc;
        }

        // Sigmoid Activation
        mxnet::Symbol SigmoidLayer(mxnet::Symbol input, string name="sigmoid") {

            mxnet::OperatorProperty *sigmoid_op =
                mxnet::OperatorProperty::Create("Activation");

            vector<pair<string, string>> sg_config;
            sg_config.push_back(make_pair("act_type","sigmoid"));
            sigmoid_op->Init(sg_config);

            vector<mxnet::Symbol> sym_vec;
            sym_vec.push_back(input);

            mxnet::Symbol sg = mxnet::Symbol::Create(sigmoid_op)(sym_vec,name);
            return sg;
        }

        // Tanh Activation
        mxnet::Symbol TanhLayer(mxnet::Symbol input, string name="tanh"){

            mxnet::OperatorProperty *tanh_op =
                mxnet::OperatorProperty::Create("Activation");

            vector<pair<string,string>> th_config;
            th_config.push_back(make_pair("act_type","tanh"));
            tanh_op->Init(th_config);

            vector<mxnet::Symbol> sym_vec;
            sym_vec.push_back(input);

            mxnet::Symbol th = mxnet::Symbol::Create(tanh_op)(sym_vec,name);
            return th;
        }

        // Configure basic two-layer neural nets
        // The 1st fully-connect with tanh act
        // The 2nd fully-connect with sigm act
        mxnet::Symbol BasicNN(string nu1, string nu2){
            mxnet::Symbol sym_x = mxnet::Symbol::CreateVariable("input");
            //mxnet::Symbol sym_fc1 = FullyConnectedLayer(sym_x, nu1, "fc1");
            //mxnet::Symbol sym_act1 = TanhLayer(sym_fc1, "act1");
            mxnet::Symbol sym_act2 = TanhLayer(sym_x, "act1");
            //mxnet::Symbol sym_fc2 = FullyConnectedLayer(sym_x, nu2, "fc2");
            //mxnet::Symbol sym_act2 = SigmoidLayer(sym_fc2, "act2");

            return sym_act2;
        }


};



/* This file was used to build the deep topic model */
int main(int argc, char ** argv) {
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
    // first config basic neural networks
    NN nn;
    mxnet::Symbol up_con = nn.BasicNN(to_string(2*K), to_string(K));
    vector<string> as = up_con.ListArguments();
    for (auto it = as.begin(); it != as.end(); ++it) {
        cout << *it << endl;
    }


    // test NDArray
    //const int m = 3;
    //const int n = 2;
    //mxnet::TShape shape = mshadow::Shape2(m,n);
    //mxnet::Context ctx = mxnet::Context::Create(mxnet::Context::kCPU, 1);
    //mxnet::NDArray a(shape, ctx, false);
    //mxnet::NDArray b(shape, ctx, false);
    //mxnet::real_t* aptr = static_cast<mxnet::real_t*>(a.data().dptr_);
    //mxnet::real_t* bptr = static_cast<mxnet::real_t*>(b.data().dptr_);
    //for (int i = 0; i < m; i++) {
    //    for (int j = 0; j < n; j++) {
    //        aptr[i*n + j] = i*n + j;
    //        bptr[i*n + j] = i*n + 2*j;
    //    }
    //}

    //mxnet::NDArray c = b;
    //c += a; // this line would cause the compiling fail
    //c.WaitToRead();
    //mxnet::real_t* cptr = static_cast<mxnet::real_t*>(c.data().dptr_);
    //cout << "NDArray value! " <<cptr[m*n - 1] << endl;


    // bind with executors

    // EM Framewrok to update






    return 0;
}

