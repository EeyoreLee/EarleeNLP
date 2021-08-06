# -*- encoding: utf-8 -*-
'''
@create_time: 2021/07/29 16:28:26
@author: lichunyu
'''

from test.test_ner_mrc import test as ner_mrc_test
from utils.flat.base import load_ner
from models.flat_bert import BertEmbedding



if __name__ == '__main__':
    # ner_mrc_test()
    dataset, vocabs, embedding = load_ner(
        '/root/hub/golden-horse/data',
        '/root/pretrain-models/flat/gigaword_chn.all.a2b.uni.ite50.vec',
        '/root/pretrain-models/flat/gigaword_chn.all.a2b.bi.ite50.vec',
        _refresh=False,
        index_token=False,
    )
    bert_embedding = BertEmbedding(vocabs['char'],model_dir_or_name='cn-wwm',requires_grad=False,word_dropout=0.01)
    pass
