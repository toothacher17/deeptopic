//#include "lda.h"

#include <iostream>

// This file was used to whether the sampler is working

/* initialize the alpha matrix */
// input is the matrix ptr, doc size M, topic size K
void init_alpha(double ** alpha_mk, int M, int K) {
    alpha_mk = new double*[M];
    for (int m = 0; m < M; m++) {
        alpha_mk[m]  =new double[K];
        for (int k = 0; k < K; k++) {
            alpha_mk[m][k] = 5.0;
        }
    }
}


/* free the alpha matrix */
void free_alpha(double ** alpha_mk, int M) {
    // size M K
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
	
    // debug first test the alpha_mk matrix
    double ** alpha_mk;
    int M = 1000; // training data size
    int K = 10;   // topic size
    init_alpha(alpha_mk, M, K);
    std::cout << alpha_mk[2][3]<< std::endl;
    std::cout << alpha_mk[902][3]<< std::endl;

    free_alpha(alpha_mk, M);

    /*
    model *lda=NULL;

	// initialize the model
    if (!(lda = model::init(argc, argv)))
	{
		show_help();
		return 1;
    }

    // Train the model
	if(lda->train())
	{
		std::cout << "Error: There exists a Bug in training part!" << std::endl;
		return 1;
	}
 
	// Finally test the model	
	if(lda->test())
	{
		std::cout << "Error: There exists a Bug in testing part!" << std::endl;
		return 1;
    }
    */
    return 0;
}



