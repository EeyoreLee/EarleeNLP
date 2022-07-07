# -*- encoding: utf-8 -*-
'''
@create_time: 2021/08/05 15:08:00
@author: lichunyu
'''

import os

from fastNLP import cache_results, Vocabulary
from fastNLP.io.loader import ConllLoader

from models.flat.flat_bert import StaticEmbedding, get_bigrams


# @cache_results(_cache_fp='cache/weiboNER_uni+bi_new', _refresh=False)
def load_ner(path,unigram_embedding_path=None,bigram_embedding_path=None,index_token=True, train_path=None, dev_path=None,
            char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0,char_word_dropout=0.01, test_path=None, \
            logger=None, with_placeholder=True, placeholder_path=None, with_test_a=False, test_a_path=None, label_word2idx=None, \
            **kwargs):

    loader = ConllLoader(['chars','target'])

    # train_path = os.path.join(path,'weiboNER_2nd_conll.train_deseg')
    # dev_path = os.path.join(path, 'weiboNER_2nd_conll.dev_deseg')
    # test_path = os.path.join(path, 'weiboNER_2nd_conll.test_deseg')

    if train_path is None:
        train_path = '/ai/223/person/lichunyu/datasets/dataf/seq_label/seq_label.train'
    if dev_path is None:
        dev_path = '/ai/223/person/lichunyu/datasets/dataf/seq_label/seq_label.test'

    # train_path = '/ai/223/person/lichunyu/datasets/dataf/seq_label/seq_label_all_all.train'
    # dev_path = '/ai/223/person/lichunyu/datasets/dataf/seq_label/seq_label_test_a_labeled.train'

    # train_path = '/ai/223/person/lichunyu/datasets/dataf/test/test_A_text.seq'
    # dev_path = '/ai/223/person/lichunyu/datasets/dataf/test/test_A_text.seq'
    # test_path = '/ai/223/person/lichunyu/datasets/tmp/test_one.txt'
    if test_path is None:
        test_path = '/ai/223/person/lichunyu/datasets/dataf/test/test_B_final_text.nonletter'

    if placeholder_path is None:
        placeholder_path = '/root/all_train.test'

    if test_a_path is None:
        test_a_path = '/ai/223/person/lichunyu/datasets/df-competition/df-511/test/test_A_text.seq'

    paths = {}
    paths['train'] = train_path
    paths['dev'] = dev_path
    paths['test'] = test_path
    paths['placeholder'] = placeholder_path
    paths['test_a'] = test_a_path

    datasets = {}

    for k,v in paths.items():
        bundle = loader.load(v)
        datasets[k] = bundle.datasets['train']

    for k,v in datasets.items():
        if logger is not None:
            logger.info('{}:{}'.format(k,len(v)))
        else:
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


    # char_vocab.from_dataset(datasets['train'],field_name='chars',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    if with_placeholder is True and with_test_a is False:
        char_vocab.from_dataset(datasets['train'],field_name='chars',no_create_entry_dataset=[datasets['dev'], datasets['placeholder']])
    elif with_placeholder is True and with_test_a is True:
        char_vocab.from_dataset(datasets['train'],field_name='chars',no_create_entry_dataset=[datasets['dev'], datasets['placeholder'], datasets['test_a']])
    else:
        char_vocab.from_dataset(datasets['train'],field_name='chars',no_create_entry_dataset=[datasets['dev']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    if label_word2idx is not None:
        label_vocab.word2idx = label_word2idx
    if logger is not None:
        logger.info('label_vocab:{}\n{}'.format(len(label_vocab),label_vocab.idx2word))

    for k,v in datasets.items():
        # v.set_pad_val('target',-100)
        v.add_seq_len('chars',new_field_name='seq_len')

    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab

    # bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    if with_placeholder is True and with_test_a is False:
        bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',no_create_entry_dataset=[datasets['dev'], datasets['placeholder']])
    elif with_placeholder is True and with_test_a is True:
        bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',no_create_entry_dataset=[datasets['dev'], datasets['placeholder'], datasets['test_a']])
        print('dataset create with test_a')
    else:
        bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',no_create_entry_dataset=[datasets['dev']])
    if index_token:
        char_vocab.index_dataset(*list(datasets.values()), field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(*list(datasets.values()),field_name='bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(*list(datasets.values()), field_name='target', new_field_name='target')

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