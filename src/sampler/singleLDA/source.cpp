#include "lda.h"

// This file was used to whether the sampler is working

/* initialize the alpha matrix */
// input is the matrix ptr, doc size M, topic size K
double** init_alpha(int M, int K) {
    double ** alpha_mk = new double* [M];
    for (int m = 0; m < M; m++) {
        alpha_mk[m] = new double[K];
        for (int k = 0; k < K; k++) {
            alpha_mk[m][k] = 1.0;
        }
    }
    return alpha_mk;
}


/* free the alpha matrix */
void free_alpha(double *** alpha_mk_ptr, int M) {
    // size M K
    double ** alpha_mk = *alpha_mk_ptr;
    if (alpha_mk) {
        for (int m = 0; m < M; m++) {
            if (alpha_mk[m]) {
                delete[] alpha_mk[m];
            }
        }
        delete[] alpha_mk;
    }
}



int main(int argc, char ** argv)
{
	
    // manually set it for test alpha
    int M = 49499; // training data size
    int K = 20;   // topic size

    // initialize the alpha matrix
    double **alpha_mk = init_alpha(M, K);

    model *lda=NULL;

	// initialize the model
    if (!(lda = model::init(argc, argv)))
	{
		//show_help();
		return 1;
    }

    // Train the model
	if(lda->train(alpha_mk))
	{
		std::cout << "Error: There exists a Bug in training part!" << std::endl;
		return 1;
	}
 
	// Finally test the model	
	if(lda->test(alpha_mk))
	{
		std::cout << "Error: There exists a Bug in testing part!" << std::endl;
		return 1;
    }

    free_alpha(&alpha_mk, M);
    return 0;
}



