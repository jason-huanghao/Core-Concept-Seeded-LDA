

def term_infos(data):
    '''
    :param data:
    :return: frequency in class
    '''
    from collections import Counter
    bows = data['bow']
    core_concepts = data['cc']
    cfs = {wid: [0] * len(core_concepts) for wid in data['dict'].id2token.keys()}
    labels = data['labels']

    for did, bow in enumerate(bows):
        if labels[did] == 'o':
            continue
        cid = core_concepts.index(labels[did])
        for wid, freq in Counter(bow).items():
            cfs[wid][cid] += freq
    return cfs


def mis_classified_term_ids(clusters, core_concepts, digit_gs):
    '''
    :param clusters: [set(word id),...]
    :param core_concepts: [cc1, ..., ccK]
    :param digit_gs: [cck: set(word id), ...]
    :return:
    1. mis_term_ids : [word id, ...]
    2. should_cc_index: [cc index, ...]
    '''
    mis_term_ids = []
    should_cc_index = []

    # from evaluation import generate_confusion_matrix
    # confuss = generate_confusion_matrix(clusters, digit_gs, core_concepts)
    # print(confuss)
    # print(sum(sum(confuss)))
    # input()

    for c_id, cc in enumerate(core_concepts):
        # mis_ids = clusters[c_id].difference(digit_gs[cc])
        mis_ids = digit_gs[cc].difference(clusters[c_id])
        mis_ids = list(mis_ids)
        mis_term_ids += mis_ids

        for wid in mis_ids:
            for c_id1, cc1 in enumerate(core_concepts):
                if c_id1 == c_id:
                    continue
                if wid in digit_gs[cc1]:
                    should_cc_index.append(c_id1)
                    break

    return mis_term_ids, should_cc_index


def mis_classified_term_info(data, mis_term_ids, all_term_ids):
    '''
    :param data:
    :param mis_term_ids:
    :return:
    1. frequency
    2. frequency in each topic / cc
    3. misclassified cc_index
    '''
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    def show_plot(x):
        x = [i for i in x if i < 40]
        x = np.array(x)
        sns.distplot(x, rug=True, hist=False)
        plt.tight_layout()
        plt.show()

    def num_word_freq(x):
        from collections import Counter
        freq_num = Counter(x)
        x = [i+1 for i in range(40)]
        y = [freq_num[i+1] if i+1 in freq_num.keys() else 0 for i in range(40)]
        x = np.array(x)
        y = np.array(y)
        plt.bar(x, y)
        plt.tight_layout()
        plt.show()

    dictionary = data['dict']
    tf = dictionary.cfs
    all_term_tf = [tf[wid] for wid in all_term_ids]
    tf_miss_classified = [tf[wid] for wid in mis_term_ids]
    tf_classified = [tf[wid] for wid in set(all_term_ids).difference(mis_term_ids)]

    show_plot(all_term_tf)
    show_plot(tf_classified)
    show_plot(tf_miss_classified)

    num_word_freq(all_term_tf)
    num_word_freq(tf_classified)
    num_word_freq(tf_miss_classified)


def dis_similar_terms(sim_flag, data):
    from tools import get_index1
    token2class = {term: cc for cc, cc_set in data['gs'].items() for term in cc_set}
    id2token = data['dict'].id2token
    ccs = [cc for cc in data['gs'].keys()]
    W = len(id2token.keys())
    for wid in range(W):
        print("(", id2token[wid], token2class[id2token[wid]], ")", end=': ')
        class_stat = {cc: 0 for cc in ccs}
        for wid2 in range(W):
            if sim_flag[get_index1(wid, wid2, W)] == -1:
                cctmp = token2class[id2token[wid2]]
                print("(", id2token[wid2], cctmp, ")", end='\t')
                class_stat[cctmp] += 1
        print('\n', class_stat, '\n')
        input()
    return

# train_paras = (1,2,3,4,500,5,6)
# train_paras = list(train_paras)
# print(train_paras[-3])

# train_paras[-3] = -10  # threshold for frequency
# train_paras += [20]
# train_paras = (e for e in train_paras)
#
# print(*train_paras)
