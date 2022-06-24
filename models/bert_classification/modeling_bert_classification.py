# -*- encoding: utf-8 -*-
'''
@create_time: 2022/06/24 10:33:51
@author: lichunyu
'''
from transformers.models.bert.modeling_bert import BertForSequenceClassification


BertClassification = BertForSequenceClassification


def init_func(model, **kwds):

    return model