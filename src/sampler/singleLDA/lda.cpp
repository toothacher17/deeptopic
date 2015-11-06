#include "lda.h"

int FTreeLDA::specific_init()
{
	// Construct trees here
	std::cout << "Initializing the Fenwich trees ..." << std::endl;
	trees = new fTree[V];
	{
		double *temp = new double[K];
		for (int w = 0; w < V; ++w)
		{
			for (int k = 0; k < K; ++k)
				temp[k] = (n_wk[w][k] + beta) / (n_k[k] + Vbeta);
			trees[w].init(K);
			trees[w].recompute(temp);
		}
	}
	return 0;
}

int FTreeLDA::sampling(int m)
{
	int kc = 0;
	for (const auto& k : n_mks[m])
	{
		nd_m[k.first] = k.second;
		rev_mapper[k.first] = kc++;
	}
	for (int n = 0; n < trngdata->docs[m]->length; ++n)
	{
		int w = trngdata->docs[m]->words[n];
		
		// remove z_ij from the count variables
		int topic = z[m][n]; int old_topic = topic;
		remove_from_topic( w, m, topic);

		// update fTree[w]
		trees[w].update(topic, (n_wk[w][topic] + beta) / (n_k[topic] + Vbeta));

		//Compute pdw
		double psum = 0;
		int i = 0;
		/* Travese all non-zero document-topic distribution */
		for (const auto& k : n_mks[m])
		{
			psum += k.second * (n_wk[w][k.first] + beta) / (n_k[k.first] + Vbeta);
			p[i++] = psum;
		}

		double u = utils::unif01() * (psum + alpha*trees[w].w[1]);

		if (u < psum)
		{
			int temp = std::lower_bound(p, p+i, u) - p;
			topic = n_mks[m][temp].first;
		}
		else
		{
			topic = trees[w].sample(utils::unif01());
		}

		// add newly estimated z_i to count variables
		add_to_topic( w, m, topic, old_topic );

		// update fTree[w]
		trees[w].update(topic, (n_wk[w][topic] + beta) / (n_k[topic] + Vbeta));

		z[m][n] = topic;
	}
	for (const auto& k : n_mks[m])
	{
		nd_m[k.first] = 0;
		rev_mapper[k.first] = -1;
	}
	return 0;	
}
