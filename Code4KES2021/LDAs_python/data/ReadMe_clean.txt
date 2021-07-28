import joblib
joblib.load("XXX.job")

cs_concept.job:

keys of cs_concept.data ['cc', 'gs', 'docs', 'labels', 'dict', 'bow']

cc: core concepts
gs: gold standard
docs: documents [[w, w ..., w], [w, w, ..., w], ..., [w, w, ..., w]]
labels: labels of documents [cc1, cc2, cc1, ..., ], where cc_i is from cc
dict: gensim Dictionary dict.token2id. dict.id2token, dict.cfs, dict.dfs
bow: bag of words [[w_id, w_id, ..., w_id], [w_id, w_id, ..., w_id], ..., [w_id, w_id, ..., w_id]]

core concepts:  ['data_structures', 'cryptography', 'software_engineering', 'computer_graphics', 'network_security', 'computer_programming', 'algorithm_design', 'operating_systems', 'distributed_computing', 'machine_learning']

# terms in gs: 2327
# term in  data_structures 323
# term in  cryptography 230
# term in  software_engineering 248
# term in  computer_graphics 369
# term in  network_security 247
# term in  computer_programming 157
# term in  algorithm_design 118
# term in  operating_systems 170
# term in  distributed_computing 167
# term in  machine_learning 298

# docs:  3645

# terms 2320 (!= 2327)




music_concept.data:
keys of cs_concept.data ['cc', 'gs', 'docs', 'labels', 'dict', 'bow']

cc: core concepts
gs: gold standard
docs: documents [[w, w ..., w], [w, w, ..., w], ..., [w, w, ..., w]]
labels: None
dict: gensim Dictionary dict.token2id. dict.id2token, dict.cfs, dict.dfs
bow: bag of words [[w_id, w_id, ..., w_id], [w_id, w_id, ..., w_id], ..., [w_id, w_id, ..., w_id]]

core concepts:   ['musician', 'album', 'genre', 'instrument', 'performance']

# terms in gs: 2872
# term in  musicians 1297
# term in  albums 484
# term in  genres 395
# term in  instruments 211
# term in  performances 485

# docs:  4458

# terms 2857 (!=2872)

