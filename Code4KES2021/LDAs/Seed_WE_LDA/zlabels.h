#pragma once
#include <vector>
using namespace std;


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
	vector<vector<int>> &seedwordtopic);


class Zlabels
{
	// Fixed Parameters
	int W;	//number of vocabularies
	int D;	//number of documents
	int K;	//number of topics
	int Iter; //number of gibbs iteration

	// Tuning Parameters
	double Alpha;	//hyperparameter of Dirichlet(alpha)
	double Beta;	//hyperparameter of Dirichlet(beta)
	double Pi;		//percentage of seed word w assigned to topic k (a prior topic) 

	// Inputs
	vector<int> Nds;			// D length of each document
	vector<vector<int>> Docs;	// D*Nd word id of each token
	//vector<vector<int>> TokenPriorFlag;	// D*Nd prior topic of each token, value in [0, 1] (value=1 means token wi is a seed word)
	vector<vector<int>> WordTopics;	// W whether word is a seed word; value in [0, K], (value=K means word is a regular word)

	// Counts
	int **nw;	// W*K count of time that word w in topic k
	int **nd;	// D*K count #token under document d in topic k
	int *nk;	// K count #token under topic k

	double *pseudoPdk;	// K pseudo P(k|d)
	double *pseudoPwk;  // K pseudo P(k|w)
	double *P;			// K P(wi=k|W(-i), alpha, beta)
	vector<vector<int>> TokenTopic;	// D*Nd topic of each token, value in [0, K-1], value=K means wi didn't be sampled yet

public:
	~Zlabels();
	void SetData(
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
		vector<vector<int>> &WordTopics
	);

	void Init();
	void MarkovChain();
	void sampling(int did, int tid);
	void train();
	vector<vector<double>> WordTopicMatrix();	// W*K
	vector<vector<double>> Phi();	// K*W
	vector<vector<double>> Theta();	// D*K
	double Perlexity();
};
