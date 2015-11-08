#ifndef _LDA_H
#define _LDA_H

#include <iostream>

#include "utils.h"
#include "fTree.h"

#include "model.h"

class FTreeLDA : public model
{
public:
	fTree *trees;

	// estimate LDA model using F+ Tree
	int specific_init();
	int sampling(int m, double ** alpha_mk);
};

#endif
