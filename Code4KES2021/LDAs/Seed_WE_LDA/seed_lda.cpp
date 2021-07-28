#include "seed_lda.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <map>
#include "tools.h"
#include <time.h>

//#include<pybind11/pybind11.h>
//#include <pybind11/stl.h>
//
//namespace py = pybind11;

using namespace std;


int iVerbose = 1;


void SeededLDA::SetData(
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
	vector<vector<int>>& doc_topics) {
	/*
	word id begins at 0,
	doc id begins at 0,
	word_topics [[topic_id, ...], [], ..., ]	topic_id begins at 0	length = len(token_wordids)
	*/
	//int SEED = 1 + (rand() % 10000);
	int SEED = 3;
	if (!SEED % 2)
		SEED += 1;
	seedMT(1 + SEED * 2); // seeding only works on uneven numbers
	srand(int(time(NULL)) + rand() % 10000);
	srand(0);

	iNoIter = Iteration;

	K = TopicNumber;
	S = K;
	W = WordNumber;
	D = DocNumber;

	Alpha = alpha;
	Beta = beta;
	WBeta = (double)(W*Beta);
	KAlpha = (double)(K*Alpha);
	Mu = mu;
	dWMu = W * Mu;
	Pi = tau;
	SPi = S * Pi;

	// fill word id & doc id
	iNoTokens = token_wordids.size();					// number of tokens
	w = new int[iNoTokens];								// word id of tokens
	d = new int[iNoTokens];								// document id of tokens
	for (int i = 0; i < iNoTokens; i++) {
		w[i] = token_wordids[i];
		d[i] = doc_ids[i];
	}
	// token & words
	z = (int *)calloc(iNoTokens, sizeof(int));			// D*Nd topic id of tokens
	tokenOrder = (int *)calloc(iNoTokens, sizeof(int));	// D*Nd token order
	bSeed = (bool *)calloc(iNoTokens, sizeof(bool));	// D*Nd seed regular flag
	//bSeedTot = (int *)calloc(2 * K, sizeof(int));		// 2*K #token in regular(even) and seed(odd)
	// regular topic
	ztot = (int *)calloc(K, sizeof(int));				// K #token in regular topic k
	wp = (int *)calloc(K*W, sizeof(int));				// K*W #time word w in regular topic k
	// seed topic
	ztotSeed = (int *)calloc(K, sizeof(int));			// K #token in seed topic k
	wpSeed = (int *)calloc(K*W, sizeof(int));			// K*W #time word w in seed topic k
	// seed topic words
	seedTopicWords.clear();
	seedTopicWords = word_topics;				// {word: set(topic id, ..., topic id)} length = |vocabularies|
	seedtot = (int *)calloc(S, sizeof(int));	// #token of each group

	// documents
	Nds = nds;
	dp = (int *)calloc(K*D, sizeof(int));		// document-topic count matrix
	s = (int *)calloc(D, sizeof(int));			// group id of documents
	docOrder = (int *)calloc(D, sizeof(int));	// document order
	seed2top = (int *)calloc(S*K, sizeof(int));	// group-topic count matrix
	seeddoc = (int *)calloc(S, sizeof(int));	// #document of each group
	// seed topic docs
	seedDocTopics.clear();
	seedDocTopics = doc_topics;					// {doc: set(topic id, ..., topic di)}	length = |Documents|

}

void SeededLDA::RandomOrder() {
	int rp, temp;
	for (int i = 0; i < iNoTokens; i++) tokenOrder[i] = i; // fill with increasing series
	for (int i = 0; i < (iNoTokens - 1); i++) {
		// pick a random integer between i and nw
		rp = i + (int)((double)(iNoTokens - i) * (double)randomMT() / (double)(4294967296.0 + 1.0));

		//rp = i + (int)((double)(iNoTokens - i) * ((double)rand() / RAND_MAX));
		// switch contents on position i and position rp
		temp = tokenOrder[rp];
		tokenOrder[rp] = tokenOrder[i];
		tokenOrder[i] = temp;
	}
	for (int i = 0; i < D; i++) docOrder[i] = i;
	for (int i = 0; i < D - 1; i++) {
		rp = i + (int)((double)(D - i) * (double)randomMT() / (double)(4294967296.0 + 1.0));

		//rp = i + (int)((double)(D - i) * (double)rand() / RAND_MAX);
		temp = docOrder[rp];
		docOrder[rp] = docOrder[i];
		docOrder[i] = temp;
	}
}

double SeededLDA::GetWordTopicProb(int iWordId, int iTopic) {
	//return (((double) wp[iWordId* K+iTopic] + Beta)/((double)ztot[iTopic]+WBeta));
	double dx, dFrac;
	int wioffset = iWordId * K;
	dx = Pi;
	wioffset = iWordId * K;
	dFrac = dx * (((double)wp[wioffset + iTopic] + Beta) / ((double)ztot[iTopic] + WBeta));
	dFrac += (1 - dx) * ((double)wpSeed[wioffset + iTopic] + (double)Mu) / ((double)ztotSeed[iTopic] + (double)dWMu);
	return dFrac;
}

double SeededLDA::GetPseduoWordTopicProb(int iWordId, int iTopic) {
	//return (((double) wp[iWordId* K+iTopic] + Beta)/((double)ztot[iTopic]+WBeta));
	double dx, dFrac;
	int wioffset = iWordId * K;
	dx = Pi;
	wioffset = iWordId * K;
	dFrac = dx * ((double)wp[wioffset + iTopic]);
	dFrac += (1 - dx) * ((double)wpSeed[wioffset + iTopic]);
	return dFrac;
}

double SeededLDA::GetTopicDocProb(int iTopic, int iDocId) {
	return (((double)dp[iDocId*K + iTopic] + Alpha) / ((double)Nds[iDocId] + KAlpha));
}

double SeededLDA::GetWordDocProb(int iWordId, int iDocId) {
	double dWordDocProb = 0;
	for (int j = 0; j < K; j++) {
		double dWordTopicProb = GetWordTopicProb(iWordId, j);
		double dTopicDocProb = GetTopicDocProb(j, iDocId);
		dWordDocProb += (dWordTopicProb * dTopicDocProb);
	}
	return dWordDocProb;
}

bool SeededLDA::CheckDocDistribution(int iDocId) {
	int dioffset = iDocId * K, iTemp = 0;
	double dTempProb = 0;
	for (int i = dioffset; i < dioffset + K; i++) {
		iTemp += dp[i];
	}
	for (int k = 0; k < K; k++)
		dTempProb += GetTopicDocProb(k, iDocId);
	if (iTemp != Nds[iDocId]) {
		cerr << "Tring Tring .... the total number of class assignments " << iTemp << " didn't match with doc length " << Nds[iDocId] << endl;
		return false;
	}
	if (!isfinite(dTempProb) || fabs(dTempProb - 1) > 1e-8) {
		fprintf(stderr, "Strange ... Though integers counted, sum p(.|d=%d) = %e didn't match\n", iDocId, dTempProb);
		return false;
	}
	return true;
}

bool SeededLDA::CheckTopicDistribution(int iTopic) {
	int iTemp = 0;
	double dTempProb = 0;
	for (int w = 0; w < W; w++) {
		iTemp += wp[w*K + iTopic];
		dTempProb += GetWordTopicProb(w, iTopic);
	}
	if (iTemp != ztot[iTopic]) {
		fprintf(stderr, "Ooops ... No dict entries assigned to p(.|k=%d) didn't match\n", iTopic);
		return false;
	}

	if (!isfinite(dTempProb) || fabs(dTempProb - 1) > 1e-8) {
		fprintf(stderr, "Strange ... Though integers counted, sum p(.|k=%d) = %e didn't match\n", iTopic, dTempProb);
		return false;
	}

	//if (iVerbose > 1)
	//	printf("Topic:%d Total seed tokens:%d (fromSeedTopic:%d fromDocTopic:%d)\n", iTopic, bSeedTot[2 * iTopic] + bSeedTot[2 * iTopic + 1], bSeedTot[2 * iTopic + 1], bSeedTot[2 * iTopic]);
	//printf("Topic:%d frmSeed:%d frmDoc:%d Total:%d\n", iTopic, ztotSeed[iTopic], ztot[iTopic], ztotSeed[iTopic]+ztot[iTopic]);
	return true;
}

bool SeededLDA::CheckConsistency() {
	bool bReturn = true;

	int iTempTot = 0, iTemp;
	for (int iDocId = 0; iDocId < D; iDocId++) {
		if (CheckDocDistribution(iDocId) == false)
			bReturn = false;
		iTemp = 0;
		for (int j = 0; j < K; j++)
			iTemp += dp[iDocId*K + j];
		if (iTemp != Nds[iDocId]) {
			fprintf(stderr, "Document length %d didn't match total tokens (%d) of this doc\n", Nds[iDocId], iTemp);
			exit(1234);
		}
		iTempTot += iTemp;
	}

	for (int topic = 0; topic < K; topic++) {
		if (CheckTopicDistribution(topic) == false)
			bReturn = false;
	}
	int iTotDocs = 0;
	for (int ts = 0; ts < S; ts++) {
		iTotDocs += seeddoc[ts];
		iTemp = 0;
		for (int j = 0; j < K; j++)
			iTemp += seed2top[ts*K + j];
		if (iTemp != seedtot[ts]) {
			fprintf(stderr, "Total tokens of seed topic %d, didn't match [%d != %d]\n", ts, iTemp, seedtot[ts]);
		}
	}
	if (iTotDocs != D) {
		fprintf(stderr, "Total number of docs %d didn't match with seedtot %d\n", D, iTotDocs);
		return false;
	}
	int *seedCountsTmp = (int *)calloc(S*K, sizeof(int));
	for (int iDocId = 0; iDocId < D; iDocId++) {
		int seedtopic = s[iDocId];
		int offset = seedtopic * K, dioffset = iDocId * K;
		for (int j = 0; j < K; j++)
			seedCountsTmp[offset + j] += dp[dioffset + j];
	}
	for (int i = 0; i < S*K; i++)
		if (seedCountsTmp[i] != seed2top[i]) {
			fprintf(stderr, "Collective counts of topic %d in seedtopic %d didn't match\n", i%K, i / K);
		}

	return bReturn;
}

double SeededLDA::Perplexity(int n, const int *w, const int *d) {
	int i, wi, di, wioffset, dioffset;
	double dWordDocProb, dprob, dPerp;
	int iSummedOver = 0;
	dPerp = 0;
	for (i = 0; i < n; i++) {
		wi = w[i];
		di = d[i];
		//if(dUnigramProbs[wi] > 4)
		//	continue;

		dWordDocProb = GetWordDocProb(wi, di);
		dPerp += log(dWordDocProb);
		iSummedOver += 1;
	}
	dPerp *= -1;
	dPerp /= (double)iSummedOver;
	return exp(dPerp);
}

void SeededLDA::RandomInitialize() {
	int i, topic, wi, di, widx, itidx, seedtopicidx, seedtopic;
	double dRand;
	// document group id initialization
	for (i = 0; i < D; i++) {				// documents
		if (seedDocTopics[i].size() > 0) {
			seedtopicidx = (int)((double)randomMT() * seedDocTopics[i].size() / (double)(4294967296.0 + 1.0));
			//seedtopicidx = (int)((double)rand() / (RAND_MAX + 1.) * (double)seedDocTopics[i].size());
			seedtopic = seedDocTopics[i][seedtopicidx];
		}
		else {
			seedtopic = (int)((double)randomMT() * (double)S / (double)(4294967296.0 + 1.0));
			//seedtopic = (int)((double)rand() / (RAND_MAX + 1.) * (double)S);		// 这个地方不应该是 S  而应该用Seed Word的组数
		}
		s[i] = seedtopic;
		seeddoc[seedtopic]++;
	}
	// token topic id initialization
	for (i = 0; i < iNoTokens; i++)		// terms D * Nd
	{
		wi = w[i];
		di = d[i];
		seedtopic = s[di];

		bool bFrmSeedTopic = false;
		if (seedTopicWords[wi].size() > 0) {
			dRand = ((double)randomMT() / (double)(4294967296.0 + 1.0));
			//dRand = ((double)rand() / (double)(RAND_MAX+1.0));
			if (dRand < Pi)
				bFrmSeedTopic = true;
		}

		bSeed[i] = bFrmSeedTopic;
		if (bFrmSeedTopic == false) {			// regular topics
			//topic = (int)((double)rand() / (RAND_MAX + 1.) * (double)K);
			topic = (int)((double)randomMT() * (double)K / (double)(4294967296.0 + 1.0));

			ztot[topic]++;
			wp[wi*K + topic]++;
			//if (seedTopicWords[wi].size() > 0)
				//bSeedTot[topic * 2]++;
		}
		else {											// seeded topics
			if (seedTopicWords[wi].size() == 1)
				topic = seedTopicWords[wi][0];
			else {
				itidx = (int)((double)randomMT() * (double)seedTopicWords[wi].size() / (double)(4294967296.0 + 1.0));

				//itidx = (int)((double)rand() / (double)(RAND_MAX + 1.) * (double)seedTopicWords[wi].size());
				topic = seedTopicWords[wi][itidx];		// jason 这里的 seedTopicWords[wi] 有可能为空vector
			}
			ztotSeed[topic]++;
			wpSeed[wi*K + topic]++;
			//bSeedTot[topic * 2 + 1]++;
		}
		if (topic < 0 || topic >= K) {
			fprintf(stderr, "sampled incorrect topic %d\n", topic);
			exit(1234);
		}
		z[i] = topic; // assign this word token to this topic
		dp[di*K + topic]++; // increment dp count matrix
		seed2top[seedtopic*K + topic]++;
		seedtot[seedtopic]++;
	}
	if (CheckConsistency() == false) {
		fprintf(stderr, "Initial topic assignments are not consistent\n");
		exit(1234);
	}
}

void SeededLDA::SampleDocSeedTopic(int di) {
	if (seedDocTopics[di].size() == 1) return;
	// sampling group id for document, to adapt alpha for a document
	int seedtopic, j, ts;
	double r, max, totprob;
	double *seedprobs = (double *)calloc(S, sizeof(double));
	bool *bConsider = (bool *)calloc(S, sizeof(bool));

	seedtopic = s[di];
	seeddoc[seedtopic]--;
	for (j = 0; j < K; j++) {
		seed2top[seedtopic*K + j] -= dp[di*K + j];

		if (seed2top[seedtopic*K + j] < 0) {
			fprintf(stderr, "Counts (k=%d|ts=%d) became %d < 0\n", j, seedtopic, seed2top[seedtopic*K + j]);
		}
	}
	seedtot[seedtopic] -= Nds[di];
	if (seeddoc[seedtopic] < 0 || seedtot[seedtopic] < 0) {
		fprintf(stderr, "Either seeddoc[%d]=%d < 0 or seedtot[%d]=%d < 0\n", seedtopic, seeddoc[seedtopic], seedtopic, seedtot[seedtopic]);
	}

	totprob = 0;
	double dprob, dmaxprob = -1 * INFINITY;
	for (ts = 0; ts < S; ts++)
		bConsider[ts] = false;
	if (seedDocTopics[di].size() > 0) {		// if doc di contains any seed word, then sample over limited seed topic
		for (ts = 0; ts < S; ts++)
			seedprobs[ts] = -1 * INFINITY;
		for (int tsidx = 0; tsidx < seedDocTopics[di].size(); tsidx++) {
			ts = seedDocTopics[di][tsidx];
			bConsider[ts] = true;

			dprob = log((double)(seeddoc[ts] + Pi)) - Nds[di] * log(seedtot[ts] + KAlpha);	// dp[d,k] / Nd[d] normailize by doc length

			int iTempDocLength = 0;
			for (j = 0; j < K; j++) {
				dprob += (double)dp[di*K + j] * log((double)seed2top[ts*K + j] + Alpha);
				iTempDocLength += dp[di*K + j];
			}
			if (iTempDocLength != Nds[di])
				fprintf(stderr, "Document lengths didn't match\n");

			if (dmaxprob < dprob)
				dmaxprob = dprob;
			seedprobs[ts] = dprob;
		}
	}
	else {											// else sample over all seed topic
		for (ts = 0; ts < S; ts++) {
			bConsider[ts] = true;
			dprob = log((double)(seeddoc[ts] + Pi)) - Nds[di] * log(seedtot[ts] + KAlpha);

			for (j = 0; j < K; j++) {
				dprob += (double)dp[di*K + j] * log((double)seed2top[ts*K + j] + Alpha);
			}
			if (dmaxprob < dprob)
				dmaxprob = dprob;
			seedprobs[ts] = dprob;
		}
	}

	/*for (ts = 0; ts < S; ts++)
		cout << seedprobs[ts] << " ";
	cout << endl;*/

	// Compute the cumulative log probabilities
	for (ts = 0; ts < S; ts++) {
		if (bConsider[ts]) {
			totprob = seedprobs[ts];
			break;
		}
	}
	for (ts = ts + 1; ts < S; ts++) {
		// log(totprob) = log(totprob)+log(xcur)
		if (bConsider[ts]) {
			totprob = dmaxprob + log(exp(totprob - dmaxprob) + exp(seedprobs[ts] - dmaxprob));
			seedprobs[ts] = totprob;
		}
	}
	r = (double)randomMT() / (double) 4294967296.0;
	//r = (double)rand() / RAND_MAX;
	r = log(r) + totprob;
	max = seedprobs[0];
	seedtopic = 0;
	while (r > max) {
		seedtopic++;
		max = seedprobs[seedtopic];
	}

	//cout << seedtopic << "\n" << endl;

	if (seedtopic < 0 || seedtopic >= S) {
		fprintf(stderr, "Choose wrong seed topic %d is not in range [0,%d)\n", seedtopic, S);
		fprintf(stderr, "r:%lf totprob:%lf\n", r, totprob);
		exit(1234);
	}
	if (seedDocTopics[di].size() > 0) {
		bool bFound = false;
		for (int tsidx = 0; tsidx < seedDocTopics[di].size(); tsidx++)
			if (seedDocTopics[di][tsidx] == seedtopic)
				bFound = true;
		if (bFound == false) {
			fprintf(stderr, "Sampled seedtopic %d is not found in the topics allowed for this doc %d -- list [", seedtopic, di);
			for (int tsidx = 0; tsidx < seedDocTopics[di].size(); tsidx++)
				fprintf(stderr, "%d ", seedDocTopics[di][tsidx]);
			fprintf(stderr, "]\n");
			exit(1234);
		}
	}

	s[di] = seedtopic;
	seeddoc[seedtopic]++;
	for (j = 0; j < K; j++)
		seed2top[seedtopic*K + j] += dp[di*K + j];
	seedtot[seedtopic] += Nds[di];

	free(seedprobs);
	free(bConsider);
}

void SeededLDA::Iterate(bool bOnTestSet, int iNoSamples, const int *iOrder, int iNoExtraIter) {
	int wi, di, i, ii, j, topic, iter, wioffset, dioffset, widx, seedtopic;
	double totprob, r, max, dTemp;
	double *probs = (double *)calloc(2 * K, sizeof(double));

	if (CheckConsistency() == false) {
		fprintf(stderr, "Initial topic assignments are not consistent\n");
		exit(1234);
	}
	PauseToEvaluate(0, bOnTestSet, iNoSamples, iOrder);
	int iRunFor = (iNoExtraIter == 0 ? iNoIter : iNoExtraIter);
	bool *bSampled = (bool *)calloc(D, sizeof(double));
	bool bFrmSeedTopic;
	for (iter = 1; iter <= iRunFor; iter++) {

		for (ii = 0; ii < D; ii++) {
			bSampled[ii] = false;
		}

		for (ii = 0; ii < iNoTokens; ii++) {
			i = tokenOrder[ii]; // current word token to assess
			wi = w[i]; // current word index
			di = d[i]; // current document index
			topic = z[i]; // current topic assignment to word token
			if (bSampled[di] == false) {
				SampleDocSeedTopic(di);
				bSampled[di] = true;
			}

			seedtopic = s[di];

			wioffset = wi * K;
			dioffset = di * K;

			bFrmSeedTopic = bSeed[i];
			dp[dioffset + topic]--;
			seed2top[seedtopic*K + topic]--;
			seedtot[seedtopic]--;

			if (bFrmSeedTopic) {
				ztotSeed[topic]--;
				wpSeed[wioffset + topic]--;
				//bSeedTot[2 * topic + 1]--;

				/*if (ztotSeed[topic] < 0 || wpSeed[wioffset + topic] < 0 || bSeedTot[2 * topic + 1] < 0) {
					fprintf(stderr, "frmSeedTopic for topic %d counts became %d, %d, %d\n", topic, ztotSeed[topic], wpSeed[wioffset + topic], bSeedTot[2 * topic + 1]);
					exit(1234);
				}*/
			}
			else {
				ztot[topic]--;  // substract this from counts
				wp[wioffset + topic]--;
				//if (seedTopicWords[wi].size() > 0) bSeedTot[2 * topic]--;

				//if (ztot[topic] < 0 || wp[wioffset + topic] < 0) {	// || bSeedTot[2 * topic] < 0
				//	fprintf(stderr, "doc counts became %d, %d, %d\n", ztot[topic], wp[wioffset + topic], bSeedTot[2 * topic]);
				//	exit(1234);
				//}
			}

			totprob = (double)0;
			if (seedTopicWords[wi].size() == 0)		// regular word wi
			{
				for (j = 0; j < K; j++) {
					double dTmpAlpha = (seed2top[seedtopic*K + j] + Alpha) / (seedtot[seedtopic] + KAlpha);
					probs[j] = ((double)wp[wioffset + j] + (double)Beta) / ((double)ztot[j] + (double)WBeta)*((double)dp[dioffset + j] + (double)dTmpAlpha);
					probs[j] *= (((double)ztot[j] + Pi) / (double)(ztot[j] + ztotSeed[j] + 2 * Pi));
					totprob += probs[j];
				}
				for (j = 0; j < K; j++)
					probs[j + K] = 0;
			}
			else {									// seed word wi 
				for (j = 0; j < K; j++) {
					double dTmpAlpha = (seed2top[seedtopic*K + j] + Alpha) / (seedtot[seedtopic] + KAlpha);
					probs[j] = ((double)wp[wioffset + j] + (double)Beta) / ((double)ztot[j] + (double)WBeta)*((double)dp[dioffset + j] + (double)dTmpAlpha);
					probs[j] *= (1 - Pi);
					totprob += probs[j];
				}
				for (j = 0; j < K; j++)
					probs[j + K] = 0;
				for (int jtmp = 0; jtmp < seedTopicWords[wi].size(); jtmp++) {
					j = seedTopicWords[wi][jtmp];
					double dTmpAlpha = (seed2top[seedtopic*K + j] + Alpha) / (seedtot[seedtopic] + KAlpha);
					probs[j + K] = ((double)wpSeed[wioffset + j] + (double)Mu) / ((double)ztotSeed[j] + (double)dWMu)*((double)dp[dioffset + j] + (double)dTmpAlpha);
					probs[j + K] *= Pi;
					totprob += probs[j + K];
				}
			}

			// sample a topic from the distribution
			r = (double)totprob * (double)randomMT() / (double) 4294967296.0;
			//r = (double)totprob * (double)rand() / RAND_MAX;
			max = probs[0]; topic = 0;

			while (r > max) {
				topic++;
				max += probs[topic];
			}

			if (topic < 0 || topic >= 2 * K) {
				printf("%lf %lf\n", r, totprob);
				fprintf(stderr, "iteration %d: sampled incorrect topic %d\n", iter, topic);
				exit(1234);
			}

			if (topic < K) {	// regular topic
				bFrmSeedTopic = false;
			}
			else {				// seed topic
				bFrmSeedTopic = true;
				topic -= K;
			}
			bSeed[i] = bFrmSeedTopic;

			if (bFrmSeedTopic) {
				bool bFound = false;
				for (j = 0; j < seedTopicWords[wi].size(); j++) {
					if (seedTopicWords[wi][j] == topic) {
						bFound = true;
						break;
					}
				}
				if (bFound == false) {
					printf("%lf %lf\n", r, totprob);
					for (j = 0; j < 2 * K; j++)
						printf("%lf ", probs[j]);
					printf("\n");
					//fprintf(stderr, "Sampled topic %d, which is not allowed for word %s\n", topic, Words[wi].c_str());
					exit(1234);
				}
			}

			z[i] = topic; // assign current word token i to topic j
			dp[dioffset + topic]++;
			seed2top[seedtopic*K + topic]++;
			seedtot[seedtopic]++;
			if (bFrmSeedTopic) {
				ztotSeed[topic]++;
				wpSeed[wioffset + topic]++;
				//bSeedTot[2 * topic + 1]++;
			}
			else {
				wp[wioffset + topic]++; // and update counts
				ztot[topic]++;
				/*if (seedTopicWords[wi].size() > 0)
					bSeedTot[2 * topic]++;*/
			}
		}
		if ((iter % 100) == 0) {
			PauseToEvaluate(iter, bOnTestSet, iNoSamples, iOrder);
		}
		if (iter % 20 == 0) {
			cout << Perplexity(iNoTokens, w, d) << endl;
		}

	}

	if (iRunFor % 100 != 0)
		PauseToEvaluate(iter, bOnTestSet, iNoSamples, iOrder);
}

void SeededLDA::PauseToEvaluate(int iter, bool bOnTestSet, int iNoSamples, const int *iOrder) {
	double dPerp, dELLikelihood;
	if (CheckConsistency() == false) {
		cerr << "Inconsistency in the probabilities " << endl;
		exit(1);
	}
	/*printf( " %d /%d, perp..." , iter , iNoIter);*/
	dPerp = Perplexity(iNoTokens, w, d);
	//printf(" (train) %lf",dPerp);

	if (iVerbose > 1) {
		for (int ts = 0; ts < S; ts++) {
			printf("doc:%d\t Tokens:%d Avg:%f\t", seeddoc[ts], seedtot[ts], (double)seedtot[ts] / seeddoc[ts]);
			for (int j = 0; j < K; j++)
				printf("%d ", seed2top[ts*K + j]);
			printf("\n");
		}
	}
}

void SeededLDA::SaveWordTopicDistribution(string sFilePath) {
	cout << "save" << sFilePath.c_str() << endl;
	ofstream out(sFilePath.c_str());
	for (int w = 0; w < W; w++) {
		for (int k = 0; k < K; k++) {
			out << GetWordTopicProb(w, k);
			if (k < K - 1)
				out << " ";
			else
				out << "\n";
			//out << k < K -1 ? " " : "\n";
		}
	}
	out.close();
}

vector<vector<double>> SeededLDA::Phi() {
	vector<vector<double>> phi;

	for (int k = 0; k < K; k++) {
		vector<double> tmp_dist;
		for (int w = 0; w < W; w++)
			tmp_dist.push_back(GetWordTopicProb(w, k));
		phi.push_back(tmp_dist);
	}
	return phi;
}

vector<vector<double>> SeededLDA::Theta() {
	vector<vector<double>> theta;
	for (int d = 0; d < D; d++) {
		vector<double> tmp_dist;
		for (int k = 0; k < K; k++)
			tmp_dist.push_back(GetTopicDocProb(k, d));
		theta.push_back(tmp_dist);
	}
	return theta;
}

vector<vector<double>> SeededLDA::WordTopicMatrix() {
	vector<vector<double>> word_topic_dist;
	for (int w = 0; w < W; w++) {
		vector<double> tmp_dist;
		for (int k = 0; k < K; k++) {
			tmp_dist.push_back(GetPseduoWordTopicProb(w, k));
		}
		word_topic_dist.push_back(tmp_dist);
	}
	return word_topic_dist;
}


void SeededLDA::train() {
	RandomInitialize();
	RandomOrder();
	Iterate(true);
}

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
	vector<vector<int>>& doc_topics) {
	cout << "Seed  LDA Training..." << endl;
	SeededLDA* glda = new SeededLDA();
	glda->SetData(token_wordids,
		doc_ids,
		nds,
		TopicNumber,
		WordNumber,
		DocNumber,
		Iteration,
		alpha,
		beta,
		mu,
		tau,
		word_topics,
		doc_topics);

	glda->RandomInitialize();
	glda->RandomOrder();
	glda->Iterate(true);
	//vector<vector<double>> result = glda->GetWordTopicDistribution();
	vector<vector<double>> result = glda->WordTopicMatrix();
	delete glda;
	return result;
}

//int main(int argc,char *argv[])
//{
//	srand(int(time(NULL))+rand()%10000);
//
//	int iTopicalWords = 30;
//	string sConfigPath, sAssignPath, sJob = "save", wordTopicDistributionPath, regularWordTopicDistributionPath;
//	string sBestTopicFilePath = "/linqshomes/artir/Doctorate/projects/mooc/topic-models/SeededLDA/src/SeededLDA_bestTopic.txt";
//	string sDocTopicDistFilePath = "/linqshomes/artir/Doctorate/projects/mooc/topic-models/SeededLDA/src/SeededLDA_docTopicDist.txt";
//	if(argc < 2){
//		printf("Usage: ./a.out config_file_path model_file_path [load/save]\n");
//		return 0;
//	}else if(argc == 2){
//		sConfigPath = argv[1];
//		printf("Running: %s %s\n", argv[0], sConfigPath.c_str());
//	}else if(argc == 3){
//		sConfigPath = argv[1];
//		sAssignPath = argv[2];
//		printf("Running: %s %s %s\n", argv[0], sConfigPath.c_str(), sAssignPath.c_str());
//	}else if(argc == 4){
//		sConfigPath = argv[1];
//		sAssignPath = argv[2];
//		sJob = argv[3];
//		printf("Running: %s %s %s %s\n", argv[0], sConfigPath.c_str(), sAssignPath.c_str(), sJob.c_str());
//	}else if(argc == 5){
//		sConfigPath = argv[1];						// config.status
//		wordTopicDistributionPath = argv[2];		// matrix.model
//		regularWordTopicDistributionPath = argv[3];	// tmp.model (regular model)
//		sJob = argv[4];								// save
//		printf("Running: %s %s %s %s %s\n", argv[0], sConfigPath.c_str(), wordTopicDistributionPath.c_str(),regularWordTopicDistributionPath.c_str(), sJob.c_str());
//	}
//
//	SeededLDA glda;
//	glda.LoadData(sConfigPath);
//
//	if(sJob == "save"){
//		glda.RandomInitialize();
//		
//		glda.RandomOrder();
//		//cout << "1.-----------------save" << endl;
//		glda.Iterate(true); //, 0, NULL, 100);
//		//cout << "-----------------save" << wordTopicDistributionPath << endl;
//		glda.SaveWordTopicDistribution(wordTopicDistributionPath);
//		glda.SaveRegularTopicDistribution(regularWordTopicDistributionPath);
//		// glda.SaveTopicAssignments(sAssignPath);
//	}
//	//glda.PrintTopicalWords(iTopicalWords);
//	//glda.PrintBestTopic(sBestTopicFilePath);
//	//glda.PrintDocTopicDist(sDocTopicDistFilePath);
//	//glda.LeftOutTokens();
//}

//PYBIND11_MODULE(Seed_WE_LDA, m) {
//	m.doc() = "train seeded LDA and get a matrix with shape of W*K using pybind11"; // optional module docstring
//	m.def("train_seed_lda", &train_seed_LDA, "Add two NumPy arrays");
//}


//void SeededLDA::EntropyOfDistributions() {
//	double dAvgEntropy = 0, dTmp;
//	for (int di = 0; di < D; di++) {
//		for (int k = 0; k < K ; k++) {
//			dTmp = GetTopicDocProb(k, di);
//			dAvgEntropy += dTmp * log(dTmp);
//		}
//	}
//	dAvgEntropy *= -1;
//	dAvgEntropy /= (double)D;
//	// printf(" Entropy (k|d): %lf ", dAvgEntropy);
//	dAvgEntropy = 0;
//	for (int k = 0; k < K ; k++) {
//		for (int wi = 0; wi < W; wi++) {
//			dTmp = GetWordTopicProb(wi, k);
//			dAvgEntropy += dTmp * log(dTmp);
//		}
//	}
//	dAvgEntropy *= -1;
//	dAvgEntropy /= (double)T;
//	// printf("(w|k): %lf", dAvgEntropy);
//}