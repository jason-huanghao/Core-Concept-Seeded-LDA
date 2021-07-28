
def load_seed_words(seed_fn):
    '''
    :param seed_fn: file name of seed words, where line i includes seed words (separated by ",") for topic i
    :return:
    '''

    seed_lines = open(seed_fn, 'r', encoding='utf-8').readlines()
    print("topic number is ", len(seed_lines))

    seed_words = []
    topic_ids = []
    for line_id, line in enumerate(seed_lines):
        seed_words.append(line[:-1].split(','))
        topic_ids.append([line_id])
    return seed_words, topic_ids


def word_seed_topic_id_construct(SS, token2id, TIDS):
    '''
    :param SS: [[word, ..., word], ..., [word, ..., word]]
    :param token2id: dict {token: id}
    :param TIDS: [[topic id, ..., topic id], ..., [topic id, ..., topic id]]
    :return:
    '''
    word_topics = [[]]*len(token2id.keys())

    SS1 = [[token2id[w] for w in ss] for ss in SS]

    for i, ss in enumerate(SS1):
        for wid in ss:
            word_topics[wid] = TIDS[i]

    return word_topics


def doc_seed_topic_id_construct(word_topics, Docs):
    doc_topics = [[]] * len(Docs)
    for doc_id, doc in enumerate(Docs):
        seed_word_in_doc = set()
        for w_id in doc:
            if word_topics[w_id] and w_id not in seed_word_in_doc:
                seed_word_in_doc.add(w_id)
                doc_topics[doc_id].extend(word_topics[w_id])
        doc_topics[doc_id] = list(set(doc_topics[doc_id]))
    return doc_topics


def check_intersect_emtpy(set_list):
    for i, set1 in enumerate(set_list):
        for j in range(i+1, len(set_list)):
            assert not set(set1) & set(set_list[j])


def corpus2dictionary(C):
    '''
    :param C: [[word, word, ...], ..., [word, word, ...]]
    :return: gensim.corpus.Dictionary
    '''
    from gensim.corpora import Dictionary
    assert type(C) is list and type(C[0]) is list

    dict = Dictionary(C)
    dict.id2token = {wid: token for token, wid in dict.token2id.items()}
    return dict


def corpus2IdCorpus(C, dict):
    '''
    :param C: C: [[word, word, ...], ..., [word, word, ...]]
    :param dict: gensim.corpus.Dictionary of C
    :return: IDC [[word id, word id, ...], ..., [word id, word id, ...]]
    '''
    return [dict.doc2idx(doc) for doc in C]


def random_mix_data(Docs, Nds):
    import random
    from multiprocessing.dummy import Pool as ThreadPool
    from multiprocessing import cpu_count

    def mix(d):
        random.shuffle(Docs[d])

    D = len(Nds)

    core_num = cpu_count()
    pool = ThreadPool(core_num-1)
    pool.map(mix, [d for d in range(D)])
    pool.close()
    pool.join()

    random.shuffle(Docs)

    Nds = [len(doc) for doc in Docs]
    return Docs, Nds


def seedwords4cc(seed_fn, topic_num_combination):
    '''
    :param seed_fn: file name of seed word ".txt"
    :param topic_num_combination: [int, ..., int], len = length of core concepts
    :return: K * len([]) [[sw, sw, ..., sw], ..., []]
    '''
    seed_lines = open(seed_fn, 'r', encoding='utf-8').readlines()
    seed_words4cc = []
    for line_id, duplicate in enumerate(topic_num_combination):
        seed_words = seed_lines[line_id][:-1].split(',')
        seed_words4cc.append(seed_words)
    return seed_words4cc


def data_influenced_by_combination(topic_num_combination, seed_words4cc, word2id, Docs, Nds):
    '''
    :param topic_num_combination:
    :param seed_words4cc:
    :param word2id: begin at 0
    :param Docs: [[wordid, wordid, ...],..., []]
    :param Nds: [Nd, Nd, ...]
    :return:
    word_topic_ids, doc_topic_ids, topic_number
    '''
    # TODO generate seed words

    topic_number = sum(topic_num_combination)

    seed_word_topics = {}
    topic_id = 0
    for cc_id, duplicate in enumerate(topic_num_combination):
        for d in range(duplicate):
            for sw in seed_words4cc[cc_id]:
                if word2id[sw] not in seed_word_topics.keys():
                    seed_word_topics[word2id[sw]] = set()
                seed_word_topics[word2id[sw]].add(topic_id)
            topic_id += 1

    # TODO construct word_topic_ids [[topic id, topic, id], [], ...]    len(word_topics) = len(word2id)
    word_topics = []
    for w_id in range(len(word2id)):
        if w_id in seed_word_topics.keys():
            word_topics.append(list(seed_word_topics[w_id]))
        else:
            word_topics.append([])

    # TODO construct doc_topic_ids [[topic id, topic, id], [], ...]   len(doc_topics) == # of documents in corpus
    doc_topics = [[] for _ in Nds]

    for doc_id, Nd in enumerate(Nds):
        for t_id in range(Nd):
            w_id = Docs[doc_id][t_id]
            if word_topics[w_id]:
                for seed_topic in word_topics[w_id]:
                    if seed_topic not in doc_topics[doc_id]:
                        doc_topics[doc_id].append(seed_topic)

    assert len(word2id) == len(word_topics)
    assert len(doc_topics) == len(Nds)

    return topic_number, word_topics, doc_topics


def get_index(row, column):
    '''
    :param row:
    :param column:
    :return: position in list (for a down traingle matrix without diagonal element)
    '''
    if row == column:
        return 0

    if column > row:
        row, column = column, row
    index = (0 + row) * (row + 1) // 2 + column - (row)
    return index


def get_index1(i, j, n):
    '''
    :param row:
    :param column:
    :return: position in list (for a up traingle matrix without diagonal element)
    '''
    if i == j:
        return 0

    if j < i:
        i, j = j, i
    index = i * (2*n-i+1)//2+j-2*i-1
    return index



