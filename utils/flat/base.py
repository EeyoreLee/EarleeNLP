# -*- encoding: utf-8 -*-
'''
@create_time: 2021/08/05 15:08:00
@author: lichunyu
'''

import os

from fastNLP import cache_results, Vocabulary
from fastNLP.io.loader import ConllLoader

from models.flat_bert import StaticEmbedding, get_bigrams


@cache_results(_cache_fp='cache/weiboNER_uni+bi', _refresh=False)
def load_weibo_ner(path,unigram_embedding_path=None,bigram_embedding_path=None,index_token=True,
                   char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0,char_word_dropout=0.01):

    loader = ConllLoader(['chars','target'])

    train_path = os.path.join(path,'weiboNER_2nd_conll.train_deseg')
    dev_path = os.path.join(path, 'weiboNER_2nd_conll.dev_deseg')
    test_path = os.path.join(path, 'weiboNER_2nd_conll.test_deseg')

    paths = {}
    paths['train'] = train_path
    paths['dev'] = dev_path
    paths['test'] = test_path

    datasets = {}

    for k,v in paths.items():
        bundle = loader.load(v)
        datasets[k] = bundle.datasets['train']

    for k,v in datasets.items():
        print('{}:{}'.format(k,len(v)))
    # print(*list(datasets.keys()))
    vocabs = {}
    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()

    for k,v in datasets.items():
        # ignore the word segmentation tag
        v.apply_field(lambda x: [w[0] for w in x],'chars','chars')
        v.apply_field(get_bigrams,'chars','bigrams')


    char_vocab.from_dataset(datasets['train'],field_name='chars',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    print('label_vocab:{}\n{}'.format(len(label_vocab),label_vocab.idx2word))

    for k,v in datasets.items():
        # v.set_pad_val('target',-100)
        v.add_seq_len('chars',new_field_name='seq_len')

    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab

    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    if index_token:
        char_vocab.index_dataset(*list(datasets.values()), field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(*list(datasets.values()),field_name='bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(*list(datasets.values()), field_name='target', new_field_name='target')

    # for k,v in datasets.items():
    #     v.set_input('chars','bigrams','seq_len','target')
    #     v.set_target('target','seq_len')

    vocabs['bigram'] = bigram_vocab

    embeddings = {}

    if unigram_embedding_path is not None:
        unigram_embedding = StaticEmbedding(char_vocab, model_dir_or_name=unigram_embedding_path,
                                            word_dropout=char_word_dropout,
                                            min_freq=char_min_freq,only_train_min_freq=only_train_min_freq,)
        embeddings['char'] = unigram_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, model_dir_or_name=bigram_embedding_path,
                                           word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets, vocabs, embeddings