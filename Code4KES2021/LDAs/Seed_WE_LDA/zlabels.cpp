#include "zlabels.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <time.h>
#include "tools.h"
using namespace std;


void Zlabels::SetData(
	int w,
	int d,
	int k,
	int iter,
	double alpha,
	double beta,
	double pi,
	vector<int> &nds,
	vector<vector<int>> &docs,
	//vector<vector<int>> &tokenpriorflag,
	vector<vector<int>> &wordtopics) {
	W = w;
	D = d;
	K = k;
	Iter = iter;
	Alpha = alpha;
	Beta = beta;
	Pi = pi;
	Nds = nds;
	Docs = docs;
	//TokenPriorFlag = tokenpriorflag;
	WordTopics = wordtopics;
	
	this->Init();
}

void Zlabels::Init() {
	int topic, wid;

	srand(int(time(NULL)) + rand() % 10000);

	nw = new int *[W];
	for (int w = 0; w < W; w++) nw[w] = new int[K];
	nd = new int *[D];
	for (int d = 0; d < D; d++) nd[d] = new int[K];
	nk = new int[K];
	init_matrix(nw, W, K, 0);
	init_matrix(nd, D, K, 0);
	init_vector(nk, K, 0);

	for (int did = 0; did < D; did++) {
		vector<int> tmp_vector;
		for (int tid = 0; tid < Nds[did]; tid++) {
			wid = Docs[did][tid];
			topic = (int)(((double)rand() / ((double)RAND_MAX + 1.)) * K);

			tmp_vector.push_back(topic);
			nk[topic]++;
			nd[did][topic]++;
			nw[wid][topic]++;
		}
		TokenTopic.push_back(tmp_vector);
	}

	pseudoPdk = new double[K];
	pseudoPwk = new double[K];
	P = new double[K];
}

Zlabels:: ~Zlabels() {
	if (nw) {
		for (int w = 0; w < W; w++) delete[] nw[w];
		delete[] nw;
	}

	if (nd) {
		for (int d = 0; d < D; d++) delete[] nd[d];
		delete[] nd;
	}

	if (nk) delete[] nk;
	if (pseudoPdk) delete[] pseudoPdk;
	if (pseudoPwk) delete[] pseudoPwk;
	if (P) delete[] P;
}

void Zlabels::MarkovChain() {
	for (int iter = 1; iter <= Iter; iter++) {
		for (int did = 0; did < D; did++) {
			for (int tid = 0; tid < Nds[did]; tid++) {
				sampling(did, tid);
			}
		}
		if (iter % 20 == 0)
			cout << "perlexity: " << Perlexity() << endl;
	}
}

void Zlabels::sampling(int did, int tid) {
	int topic = TokenTopic[did][tid];
	int wid = Docs[did][tid];
	int k, t;
	bool seedtopicflag;
	double seed_weight;

	nw[wid][topic] --;
	nd[did][topic]--;
	nk[topic]--;

	// psedudo p(k|d)
	for (k = 0; k < K; k++)  pseudoPdk[k] = 0;
	for (k = 0; k < K; k++) {
		pseudoPdk[k] = nd[did][k] + Alpha;	// the denominator for all topic k is the same, so neglect
	}

	//---------------------------------------------------------------

	// psedudo p(w|k)
	for (k = 0; k < K; k++) pseudoPwk[k] = 0;
	for (k = 0; k < K; k++) {
		pseudoPwk[k] = (nw[wid][k] + Beta) / (nk[k] + K*Beta);
	}

	//----------------------------------------------------------
	//Z-labels topic sampling for seed words, each sead word has only one prior label
	double sum_norm = 0.0;
	for (k = 0; k < K; k++) {
		P[k] = pseudoPdk[k] * pseudoPwk[k];
		seedtopicflag = false;
		if (WordTopics[wid].size() > 0)
			for (t = 0; t < WordTopics[wid].size(); t++)
				if (k == WordTopics[wid][t]) {
					seedtopicflag = true;
					break;
				}
		if (!seedtopicflag) P[k] *= (1 - Pi);
		sum_norm += P[k];
	}

	topic = mult_sample(P, sum_norm);
	
	nw[wid][topic] ++;
	nd[did][topic] ++;
	nk[topic]++;

	TokenTopic[did][tid] = topic;
}


vector<vector<double>> Zlabels::WordTopicMatrix() {
	vector<vector<double>> word_topic_matrix;
	for (int w = 0; w < W; w++) {
		vector<double> topic_count;
		for (int k = 0; k < K; k++) {
			topic_count.push_back(nw[w][k]);
		}
		word_topic_matrix.push_back(topic_count);
	}
	return word_topic_matrix;
}

vector<vector<double>> Zlabels::Phi() {
	vector<vector<double>> phi;
	double WBeta = W * Beta;
	for (int k = 0; k < K; k++) {
		vector<double> word_dist;
		for (int w = 0; w < W; w++) {
			word_dist.push_back((nw[w][k] + Beta) / (nk[k] + WBeta));
		}
		phi.push_back(word_dist);
	}
	return phi;
}

vector<vector<double>> Zlabels::Theta() {
	vector<vector<double>> theta;
	double KAlpha = K * Alpha;
	for (int d = 0; d < D; d++) {
		vector<double> topic_dist;
		for (int k = 0; k < K; k++) {
			topic_dist.push_back((nd[d][k] + Alpha) / (Nds[d] + KAlpha));
		}
		theta.push_back(topic_dist);
	}
	return theta;
}

double Zlabels::Perlexity() {
	vector<vector<double>> phi = Phi();
	vector<vector<double>> theta = Theta();
	int total_token = 0, w = 0;
	double perplexity = 0.0, token_prob;

	for (int d = 0; d < D; d++) {
		total_token += Nds[d];
		for (int t = 0; t < Nds[d]; t++) {
			w = Docs[d][t];
			token_prob = 0.0;
			for (int k = 0; k < K; k++)
				token_prob += theta[d][k] * phi[k][w];
			perplexity += log(token_prob);
		}
	}
	perplexity *= -1;
	return exp(perplexity / total_token);
}


void Zlabels::train() {
	MarkovChain();
}

vector<vector<double>> train_Zlabels(
	int w,
	int d,
	int k,
	int iter,
	double alpha,
	double beta,
	double pi,
	vector<int> &nds,
	vector<vector<int>> &docs,
	//vector<vector<int>> &tokenpriorflag,
	vector<vector<int>> &wordtopics) {
	cout << "Zlabels Training..." << endl;
	srand(int(time(NULL)) + rand() % 10000);
	Zlabels* zlabel = new Zlabels();

	zlabel->SetData(
		w,
		d,
		k,
		iter,
		alpha,
		beta,
		pi,
		nds,
		docs,
		//tokenpriorflag,
		wordtopics);
	
	zlabel->MarkovChain();
	// cout << "Zlabels Training 2..." << endl;
	//vector<vector<double>> result = zlabel->Phi();
	vector<vector<double>> result = zlabel->WordTopicMatrix();
	delete zlabel;
	//print_matrix(result);
	return result;
}

//PYBIND11_MODULE(Seed_WE_LDA, m) {
//	m.doc() = "train seeded LDA and get a matrix with shape of W*K using pybind11"; // optional module docstring
//	m.def("train_zlabels", &train_Zlabels, "Add two NumPy arrays");
//}