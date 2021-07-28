from lda_loader import lda, zlabel, seed_lda
from tools import get_index1, corpus2dictionary, corpus2IdCorpus, random_mix_data, check_intersect_emtpy, word_seed_topic_id_construct, doc_seed_topic_id_construct
import numpy as np


'''
##################################################################################
                                All Inputs

have reference on https://guidedlda.readthedocs.io/en/latest/, we defined the inputs as:

1. C [[word, word, ...], ..., [word, word, ...]]： a corpus, where each document is a word list.
2. a dictionary Dict (gensim.corpora.dictionary) of C. (remember to generate id2token yourself)
3. IDC [[word id, word id, ...], ..., [word id, word id, ...]]: a word id version of C.
4. CC [core concept, ..., core concept]: a list of S concepts.
5. SS [[seed word, ..., seed word], ..., [seed word, ..., seed word]]: a list of S seed sets for all seeded topics.
6. TIDS [[topic id, ..., topic id], ..., [topic id, ..., topic id]]: a list of S topic id sets for all seeded topics.
   (1) the TID_s is the topic id of all terms from SS_s.
   (2) The intersection of each two topic id sets are empty.
   (3) Union of all topic id sets is set(0, 1, ..., K-1), where K is the topic number of LDA.
##################################################################################
'''


class LDA:
    '''
    ##################################################################################
                                Inputs for LDA
    1. Parameters:
        a. K: number of topics
        b. alpha: hyper-parameter for document-topic distribution
        c. beta: hyper-parameter for topic-word distribution
        d. iteration of gibbs sampling
    2. Data:
        a. X: list of list of words
        b. dictionary: gensim.corpus.dictionary from X (optional)
        c. Docs: list of list of word ids (optional)
    ##################################################################################
    '''
    def __init__(self, K, alpha, beta, iteration=100):
        '''
        :param K: number of topics
        :param alpha: alpha
        :param beta: beta
        :param iteration: iteration of gibbs sampling
        '''
        self.W = -1
        self.D = -1
        self.K = K
        self.iter = iteration
        self.alpha = alpha
        self.beta = beta
        self.Nds = None
        self.Docs = None

        self.dict = None
        self.lda_model = None

    def fit(self, X, dictionary=None, Docs=None):
        '''
        :param dictionary: gensim.corpus.dictionary
        :param Docs: [[word id, ..., word id], ..., [word id, ..., word id]]
        :param X : documents [[word, word, ...], ..., [word, word, ...]]
        :return: training
        '''
        self.dict = corpus2dictionary(C=X) if dictionary is None else dictionary
        if len(self.dict.id2token.keys()) == 0:
            self.dict.id2token = {wid: w for w, wid in self.dict.token2id.items()}
        self.Docs = corpus2IdCorpus(C=X, dict=self.dict) if Docs is None else Docs    # word ids
        self.D = self.dict.num_docs
        self.W = len(self.dict.id2token.keys())
        self.Nds = [len(doc) for doc in self.Docs]

        self.Docs, self.Nds = random_mix_data(self.Docs, self.Nds)
        self.train()

    def train(self):
        train_data = (self.W, self.D, self.K, self.iter, self.alpha, self.beta, self.Nds, self.Docs)
        self.lda_model = lda()
        self.lda_model.setData(*train_data)
        self.lda_model.train()

    def phi(self):
        return np.array(self.lda_model.phi())

    def theta(self):
        return np.array(self.lda_model.theta())

    def word_topic_matrix(self):
        return np.array(self.lda_model.WordTopicMatrix())


class Z_labels:
    '''
    ##################################################################################
                                Inputs for Z-labels
    1. Parameters:
        a. K: number of topics
        b. alpha: hyper-parameter for document-topic distribution
        c. beta: hyper-parameter for topic-word distribution
        d. pi: the confidence of topic label of seed terms
        e. iteration of gibbs sampling
    2. Data:
        a. X: list of list of words
        b. dictionary: gensim.corpus.dictionary from X (optional)
        c. Docs: list of list of word ids (optional)
        d. SS: a list of seed sets (each contains a list of words from X)
        e. TIDS: a list of topic id sets (each contains a list of topic ids)
                 TIDS_s contains all topic ids for SS_s
    ##################################################################################
    '''
    def __init__(self, K, alpha, beta, pi, iteration=100):
        '''
        :param K: number of topics
        :param alpha: alpha
        :param beta: beta
        :param iteration: iteration of gibbs sampling
        '''
        self.W = -1
        self.D = -1
        self.K = K
        self.iter = iteration
        self.alpha = alpha
        self.beta = beta
        self.pi = pi
        self.Nds = None
        self.Docs = None

        self.dict = None
        self.lda_model = None
        self.word_topics = None

    def fit(self, X, SS, TIDS, dictionary=None, Docs=None):
        '''
        :param TIDS: [[topic id, ..., topic id], ..., [topic id, ..., topic id]], where each topic id from [0, K)
        :param SS: [[word, ..., word], ..., [word, ..., word]], where each word is from X
        :param dictionary: gensim.corpus.dictionary
        :param Docs: [[word id, ..., word id], ..., [word id, ..., word id]]
        :param X : documents [[word, word, ...], ..., [word, word, ...]]
        :return: training
        '''

        self.dict = corpus2dictionary(C=X) if dictionary is None else dictionary
        if len(self.dict.id2token.keys()) == 0:
            self.dict.id2token = {wid: w for w, wid in self.dict.token2id.items()}
        self.Docs = corpus2IdCorpus(C=X, dict=self.dict) if Docs is None else Docs  # word ids
        self.D = self.dict.num_docs
        self.W = len(self.dict.id2token.keys())
        self.Nds = [len(doc) for doc in self.Docs]

        assert set([w for ss in SS for w in ss]).issubset(set(self.dict.token2id.keys()))
        assert set([i for i in range(self.K)]).issuperset(set([e for ls in TIDS for e in ls]))
        check_intersect_emtpy(SS)  # no intersection allowed between each two word sets
        check_intersect_emtpy(TIDS)  # no intersection allowed between each two topic id sets
        self.word_topics = word_seed_topic_id_construct(SS, self.dict.token2id, TIDS)

        self.Docs, self.Nds = random_mix_data(self.Docs, self.Nds)
        self.train()

    def train(self):
        train_data = (self.W, self.D, self.K, self.iter, self.alpha, self.beta, self.pi, self.Nds, self.Docs, self.word_topics)
        self.lda_model = zlabel()
        self.lda_model.setData(*train_data)
        self.lda_model.train()

    def phi(self):
        return np.array(self.lda_model.phi())

    def theta(self):
        return np.array(self.lda_model.theta())

    def word_topic_matrix(self):
        return np.array(self.lda_model.WordTopicMatrix())


class Seeded_LDA:
    '''
    ##################################################################################
                                Inputs for Seeded LDA
    1. Parameters:
        a. K: number of topics
        b. alpha: hyper-parameter for document-topic distribution
        c. beta: hyper-parameter for topic-word distribution
        d. pi: the confidence of topic label of seed terms
        e. mu: the hyper-parameter of seed topic-word distribution
        f. iteration of gibbs sampling
    2. Data:
        a. X: list of list of words
        b. dictionary: gensim.corpus.dictionary from X (optional)
        c. Docs: list of list of word ids (optional)
        d. SS: a list of seed sets (each contains a list of words from X)
        e. TIDS: a list of topic id sets (each contains a list of topic ids)
                 TIDS_s contains all topic ids for SS_s
    ##################################################################################
    '''

    def __init__(self, K, alpha, beta, pi, mu=1e-7, iteration=100):
        '''
        :param K: number of topics
        :param alpha: alpha
        :param beta: beta
        :param iteration: iteration of gibbs sampling
        '''
        self.W = -1
        self.D = -1
        self.K = K
        self.iter = iteration
        self.alpha = alpha
        self.beta = beta
        self.pi = pi
        self.mu = mu
        self.Nds = None
        self.Docs = None

        self.dict = None
        self.lda_model = None
        self.word_topics = None
        self.doc_topics = None

    def fit(self, X, SS, TIDS, dictionary=None, Docs=None):
        '''
        :param TIDS: [[topic id, ..., topic id], ..., [topic id, ..., topic id]], where each topic id from [0, K)
        :param SS: [[word, ..., word], ..., [word, ..., word]], where each word is from X
        :param dictionary: gensim.corpus.dictionary
        :param Docs: [[word id, ..., word id], ..., [word id, ..., word id]]
        :param X : documents [[word, word, ...], ..., [word, word, ...]]
        :return: training
        '''

        self.dict = corpus2dictionary(C=X) if dictionary is None else dictionary
        if len(self.dict.id2token.keys()) == 0:
            self.dict.id2token = {wid: w for w, wid in self.dict.token2id.items()}
        self.Docs = corpus2IdCorpus(C=X, dict=self.dict) if Docs is None else Docs  # word ids
        self.D = self.dict.num_docs
        self.W = len(self.dict.id2token.keys())
        self.Nds = [len(doc) for doc in self.Docs]

        assert set([w for ss in SS for w in ss]).issubset(set(self.dict.token2id.keys()))
        assert set([i for i in range(self.K)]).issuperset(set([e for ls in TIDS for e in ls]))
        check_intersect_emtpy(SS)  # no intersection allowed between each two word sets
        check_intersect_emtpy(TIDS)  # no intersection allowed between each two topic id sets
        self.word_topics = word_seed_topic_id_construct(SS, self.dict.token2id, TIDS)
        self.doc_topics = doc_seed_topic_id_construct(self.word_topics, self.Docs)

        self.Docs, self.Nds = random_mix_data(self.Docs, self.Nds)
        self.train()

    def train(self):
        word_ids = []
        doc_ids = []
        nds = []
        for did, doc in enumerate(self.Docs):
            nds.append(len(doc))
            for wid in doc:
                word_ids.append(wid)
                doc_ids.append(did)

        train_data = (word_ids, doc_ids, nds, self.K, self.W, self.D, self.iter, self.alpha, self.beta, self.mu, self.pi, self.word_topics, self.doc_topics)
        # train_data = (self.W, self.D, self.K, self.iter, self.alpha, self.beta, self.pi, self.mu, self.Nds, self.Docs,
        #                self.doc_topics, self.word_topics)

        self.lda_model = seed_lda()
        self.lda_model.setData(*train_data)
        self.lda_model.train()

    def phi(self):
        return np.array(self.lda_model.phi())

    def theta(self):
        return np.array(self.lda_model.theta())

    def word_topic_matrix(self):
        return np.array(self.lda_model.WordTopicMatrix())


# def test_lda():
#     lda_model = LDA(K=3, alpha=0.1, beta=0.01, iteration=20)
#     docs = [["1", "2", "1", "2", "1"], ["1", "2", "1", "3"], ["2", "1", "1"], ["2", "1", "3", "2", "3"]]
#     lda_model.fit(X=docs)
#     print(lda_model.phi())
#
#
# def test_zlabel():
#     lda_model = Z_labels(K=3, alpha=0.1, beta=0.01, pi=0.7, iteration=20)
#     docs = [["1", "2", "1", "2", "1"], ["1", "2", "1", "3"], ["2", "1", "1"], ["2", "1", "3", "2", "3"]]
#     SS = [["1"], ["2"], ["3"]]
#     TIDS = [[0], [1], [2]]
#     lda_model.fit(X=docs, SS=SS, TIDS=TIDS)
#     print(lda_model.phi())
#
#
# def test_seeded_lda():
#     lda_model = Seeded_LDA(K=3, alpha=0.1, beta=0.01, pi=0.7, iteration=20)
#     docs = [["1", "2", "1", "2", "1"], ["1", "2", "1", "3"], ["2", "1", "1"], ["2", "1", "3", "2", "3"]]
#     SS = [["1"], ["2"], ["3"]]
#     TIDS = [[0], [1], [2]]
#     lda_model.fit(X=docs, SS=SS, TIDS=TIDS)
#     print(lda_model.phi())


# test_lda()
# test_zlabel()
# test_seeded_lda()


