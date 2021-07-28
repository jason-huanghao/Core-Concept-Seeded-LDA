import numpy as np
from collections import Counter
from gensim.models.coherencemodel import CoherenceModel


def cluster_by_topic_probability_max(matrix):
    '''
    :param matrix: word-topic distribution: np array N * K
    :return: term clusters for topics: list of K set [set(term id, ..., ...), ..., ]
    method description: distribute term in the topic with highest topic probability
    '''
    # global term_id_remove_from_clusters

    threshold = 0
    topic_id4term = np.argmax(matrix, axis=1)
    clusters = [set() for i in range(matrix.shape[1]*2)]

    for word_id, topic_id in enumerate(topic_id4term.tolist()):

        if matrix[word_id][topic_id] > threshold:
            clusters[topic_id].add(word_id)
        else:
            clusters[matrix.shape[1]+topic_id].add(word_id)
    clusters = [t_set for t_set in clusters if t_set]
    return clusters


def generate_confusion_matrix(clusters, digit_gs, core_concepts):
    '''
    :param clusters: term clusters for topics: list of K set [set(term id, ..., ...), ..., ]
    :param digit_gs: gold standard: {cc: {w_id}, ..., } cc from 0 to 10
    :param core_concepts: [cc1, cc2, ..., ]
    :return: confusion matrix: shape= CC number * cluster number
    last raw is belongs to "Others" class
    '''

    C = len(core_concepts)
    K = len(clusters)

    confusion_matrix = np.zeros((C, K), dtype=int)

    for c_id, cc in enumerate(core_concepts):
        for k_id, cluster in enumerate(clusters):
            confusion_matrix[c_id][k_id] = len(digit_gs[cc].intersection(cluster))

    return confusion_matrix


def micro_correct_number(confusion_matrix, cc_index=None):
    '''
    :param cc_index: cc index of each cluster
    :param confusion_matrix: shape= 11 * K
    :return: micro precision: all correct number , all but not include "Others"
    micro Precision = micro Recall = micro F1
    '''
    if cc_index is not None:
        assert confusion_matrix.shape[1] == len(cc_index)
    else:
        cc_index = np.argmax(confusion_matrix, axis=0)
    correct_number_no_others = np.sum([confusion_matrix[cc][k] for k, cc in enumerate(cc_index)])
    return correct_number_no_others, cc_index


def micro_criteria(confusion_matrix, cc_index=None):
    '''
    :param confusion_matrix: shape of |CCs|*K (Others not in CCs)
    :param cc_index: shape of K, cc index of each topic (or cluster)
    :return:
    P = R
    '''
    correct_number, cc_index = micro_correct_number(confusion_matrix, cc_index=cc_index)
    return correct_number / np.sum(confusion_matrix)


def macro_criteria(confusion_matrix):
    '''
    :param confusion_matrix: shape= 11 * K
    :return: marco precision: correct cluster number / K
    '''
    cc_index = np.argmax(confusion_matrix, axis=0)

    pre_data = {c_id: {'count': 0, 'C': sum(confusion_matrix[c_id]), 'cluster': 0} for _, c_id in gs_label2id.items()}
    for k, cc in enumerate(cc_index):
        pre_data[cc]['count'] += confusion_matrix[cc][k]
        pre_data[cc]['cluster'] += sum(confusion_matrix[:, k])

    P = R = F1 = 0
    for cc, cc_data in pre_data.items():
        if cc_data['count'] == 0:
            continue
        P += cc_data['count'] / cc_data['cluster']
        R += cc_data['count'] / cc_data['C']
        F1 += 0 if P+R == 0 else 2*P*R / (P+R)

    CC_NUM = len(pre_data.keys())
    other_digit_label = gs_label2id['o']    # the digit label of “Others” class
    P_no_others = P - 0 if pre_data[other_digit_label]['cluster'] == 0 else (pre_data[other_digit_label]['count'] / pre_data[other_digit_label]['cluster'])
    R_no_others = R - 0 if pre_data[other_digit_label]['C'] == 0 else pre_data[other_digit_label]['count'] / pre_data[other_digit_label]['C']
    F1_no_others = 0 if P_no_others+R_no_others == 0 else 2*P_no_others*R_no_others / (P_no_others+R_no_others)

    return P/CC_NUM, R/CC_NUM, F1/CC_NUM, P_no_others/(CC_NUM-1), R_no_others/(CC_NUM-1), F1_no_others/(CC_NUM-1)


def silhouette_score(matrix):
    '''
    :param matrix: word-topic distribution: np array N * K
    :return:
    '''
    from sklearn.cluster import KMeans
    from sklearn import metrics
    X = matrix
    K = matrix.shape[1]
    km = KMeans(
        n_clusters=K, init='k-means++',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(X)
    cluster_labels = km.fit_predict(X)
    return metrics.silhouette_score(X, cluster_labels)


def adjust_rand_index(confusion_matrix):
    '''
    :param confusion_matrix: shape= 11 * K
    :return: adjust_rand_index

    method description:
        reference https://en.wikipedia.org/wiki/Rand_index
    '''

    def combination_over2(n):
        return n * (n - 1) / 2

    a = np.sum(confusion_matrix, axis=1)
    b = np.sum(confusion_matrix, axis=0)
    n = np.sum(a)

    sum_a = np.sum([combination_over2(ai) for ai in a])
    sum_b = np.sum([combination_over2(bj) for bj in b])
    a_b_n = sum_a * sum_b / combination_over2(n)
    sum_n_i_j = np.sum([combination_over2(n_i_j) for n_i_j in confusion_matrix.flatten()])
    sum_a_b = sum_a + sum_b

    return (sum_n_i_j - a_b_n) / (0.5 * sum_a_b - a_b_n)


def topic_coherence(data, matrix, topn=20, way=0):
    '''
    param:CC_NUM: number of Core Concepts
    :param corpus: [[(w_id, freq), (w_id, freq), ..., (w_id, freq)], [], ..., []]
    :param dictionary:
    :param id2word:
    :param cc_index: cc index of each topic (or cluster)
    :param matrix: W*K matrix
    :param topn: the top n important words of each topic
    :param way: 0 - group by core concepts, 1 - group clusters
    :return:
    # TODO this method must be putted into main program
    '''

    core_concepts = data['cc']
    CC_NUM = len(core_concepts)
    dictionary = data['dict']
    id2word = dictionary.id2token

    # topn is 20 usually
    clusters = cluster_by_topic_probability_max(matrix)
    cc_index = [i for i in range(CC_NUM)]
    assert len(clusters) == CC_NUM

    # sort clusters based on words' topic probability
    for c_id, cluster in enumerate(clusters):
        clusters[c_id] = [w_id for w_id, _ in sorted([(w_id, matrix[w_id][c_id]) for w_id in cluster], key=lambda d:d[1], reverse=True)[:topn]]

    if way == 0:
        topics = [[] for i in range(CC_NUM)]
        for c_id, cluster in enumerate(clusters):
            topics[cc_index[c_id]] += [id2word[wid] for wid in cluster]
    elif way == 1:
        topics = [[id2word[wid] for wid in cluster] for cluster in clusters]

    texts = [[id2word[word_id] for word_id in doc] for doc in data['bow']]
    corpus = [[(wid, freq) for wid, freq in Counter(doc).items()] for doc in data['bow']]

    cm = CoherenceModel(topics=topics, texts=texts, corpus=corpus, dictionary=dictionary, coherence='c_v', topn=topn, processes=0)
    return cm.get_coherence()


def evalutations(matrix, data, supervised):
    '''
    :param paras: [micro_precision, topic_coherence]:  1 is on, otherwise 0 off
    :param data:
    :param matrix: W * K
    :param supervised: true or false
    :return:
    '''

    evaluation_fun = [micro_criteria, topic_coherence]
    result = {}
    # for idx, fun in enumerate(evaluation_fun):
    #     if paras[idx]:
    #         result[idx] = fun()

    word2id = data['dict'].token2id
    core_concepts = data['cc']
    gs = data['gs']
    digital_gs = {cc: set([word2id[w] for w in gs_set if w in word2id.keys()]) for cc, gs_set in gs.items() if cc != 'o'}

    if not len(matrix[0]) == len(core_concepts) == len(digital_gs.keys()):
        print("matrix", len(matrix[0]), "\tcc", len(core_concepts), "\tgs", len(digital_gs.keys()))
        input()
    clusters = cluster_by_topic_probability_max(matrix)

    confusion_matrix = generate_confusion_matrix(clusters, digital_gs, core_concepts)

    if confusion_matrix.shape[1] != len(core_concepts):
        print("confusion matrix shape", confusion_matrix.shape, "\t", len(core_concepts))
    return micro_criteria(confusion_matrix, cc_index=[i for i in range(len(core_concepts))] if supervised else None)

