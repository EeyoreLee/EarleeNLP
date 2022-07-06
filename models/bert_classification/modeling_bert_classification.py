# -*- encoding: utf-8 -*-
'''
@create_time: 2022/06/24 10:33:51
@author: lichunyu
'''
from transformers.models.bert.modeling_bert import BertForSequenceClassification, BertConfig


BertClassification = BertForSequenceClassification


def init_func(model: BertClassification, pretrained_model_name_or_path, num_labels, **kwds):
    config = BertConfig.from_pretrained(pretrained_model_name_or_path, num_labels=num_labels)
    model = model.from_pretrained(pretrained_model_name_or_path, config=config)
    return model