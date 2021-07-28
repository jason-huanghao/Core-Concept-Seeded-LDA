from lda_wrapper import LDA, Z_labels, Seeded_LDA
from tools import load_seed_words
from evaluation import evalutations
import os
import joblib


def run_lda(K, alpha, beta, iteration, docs, dictionary, dig_docs):
    lda_model = LDA(K=K, alpha=alpha, beta=beta, iteration=iteration)
    lda_model.fit(X=docs, dictionary=dictionary, Docs=dig_docs)
    return lda_model.phi().T


def run_zlabel(K, alpha, beta, pi, iteration, docs, SS, TIDS, dictionary, dig_docs):
    lda_model = Z_labels(K=K, alpha=alpha, beta=beta, pi=pi, iteration=iteration)
    lda_model.fit(X=docs, SS=SS, TIDS=TIDS, dictionary=dictionary, Docs=dig_docs)
    return lda_model.phi().T


def run_new_seeded_lda(K, alpha, beta, pi, iteration, docs, SS, TIDS, dictionary, dig_docs):
    lda_model = Seeded_LDA(K=K, alpha=alpha, beta=beta, pi=pi, iteration=iteration)
    lda_model.fit(X=docs, SS=SS, TIDS=TIDS, dictionary=dictionary, Docs=dig_docs)
    return lda_model.phi().T


data_dir = '../data'
domain = ['cs', 'music']
clean_nosisy = ['clean', 'noise']

d = 0
cn = 1
embed_choice = 0
data_fn = os.path.join(data_dir, domain[d]+"_concept_"+clean_nosisy[cn]+'.job')
seed_fn = os.path.join(data_dir, domain[d]+"_seed_words.txt")
data = joblib.load(data_fn)


docs = data['docs']
dictionary = data['dict']
digit_docs = data['bow']
SS, TIDS = load_seed_words(seed_fn)

parameters = [10, 0.7, 0.01, 0.7, 100, docs, SS, TIDS, dictionary, digit_docs]

for i in range(10):
    matrix = run_new_seeded_lda(*parameters)
    mic_p = evalutations(matrix, data, True)
    print(mic_p, end='\n\n')



