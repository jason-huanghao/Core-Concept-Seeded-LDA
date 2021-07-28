#pragma once
#include <vector>
using namespace std;

vector<vector<double>> train_seed_LDA(
	vector<int>& token_wordids,
	vector<int>& doc_ids,
	vector<int> &nds,
	int TopicNumber,
	int WordNumber,
	int DocNumber,
	int Iteration,
	double alpha,
	double beta,
	double mu,
	double tau,
	vector<vector<int>>& word_topics,
	vector<vector<int>>& doc_topics);

class SeededLDA {
	int *z, *s, *tokenOrder, *docOrder, *ztot, *wp, *dp, *seed2top, *seedtot, *seeddoc;
	int *w, *d;
	int W, D, S, K, iNoTokens;
	double Alpha, Beta, KAlpha, WBeta;

	//SeededLDAConfig ldaConfig;
	vector<int> Nds;	// D length of each documnent
	vector<vector<int> > seedDocTopics;
	vector<vector<int> > seedTopicWords;
	vector<vector<int> > seedDocTokens;
	int *wpSeed, *ztotSeed;
	//int *bSeedTot;	// number of word in regular topic (even position) and seed topic (odd position)
	bool *bSeed;
	double Mu, Pi, dWMu, SPi;

	void Init() {
		z = NULL; s = NULL; tokenOrder = NULL; docOrder = NULL; ztot = NULL; wp = NULL; dp = NULL; seed2top = NULL; seedtot = NULL; seeddoc = NULL;
		w = NULL; d = NULL;
		W = 0; D = 0; S = 0; K = 0; iNoTokens = 0;
		iNoIter = 0;
		Alpha = 0; Beta = 0; KAlpha = 0; WBeta = 0;
	}

public:
	int iNoIter;

	SeededLDA() { Init(); }

	void SetData(
		vector<int>& token_wordids,
		vector<int>& doc_ids,
		vector<int> &nds,
		int TopicNumber,
		int WordNumber,
		int DocNumber,
		int Iteration,
		double alpha,
		double beta,
		double mu,
		double tau,
		vector<vector<int>>& word_topics,
		vector<vector<int>>& doc_topics);

	void RandomInitialize();
	void RandomOrder();
	void SampleDocSeedTopic(int di);
	void Iterate(bool bOnTestSet = true, int iNoSamples = 0, const int *iOrder = NULL, int iNoExtraIter = 0);

	double GetTopicDocProb(int iTopic, int iDocId);
	double GetWordTopicProb(int iWordId, int iTopic);
	double GetPseduoWordTopicProb(int iWordId, int iTopic);
	double GetWordDocProb(int iWordId, int iDocId);

	bool CheckTopicDistribution(int iTopic);
	bool CheckDocDistribution(int iDocId);
	bool CheckConsistency();

	double Perplexity(int iNoTokens, const int *w, const int *d);
	void PauseToEvaluate(int iter, bool bOnTestSet = true, int iNoSamples = 0, const int *iOrder = NULL);

	void SaveWordTopicDistribution(string sFilePath);
	void train();

	vector<vector<double>> Phi();
	vector<vector<double>> Theta();
	vector<vector<double>> WordTopicMatrix();

};
