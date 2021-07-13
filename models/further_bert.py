# -*- encoding: utf-8 -*-
'''
@create_time: 2021/07/07 09:52:48
@author: lichunyu
'''

from transformers import BertForMaskedLM, BertForNextSentencePrediction, BertConfig, BertTokenizerFast, BertTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn


tokenizer = BertTokenizer.from_pretrained(
    '/root/pretrain-models/bert-base-chinese'
)


mlm_model = BertForMaskedLM.from_pretrained(
    '/root/pretrain-models/bert-base-chinese'
)

nsp_model = BertForNextSentencePrediction.from_pretrained(
    '/root/pretrain-models/bert-base-chinese'
)


"""
10% 不变
10% 换成错误的
80% MASK
"""


a = "收获颇数。算是作者的同行，最近也在做类似[MASK]东西，但是作者都给总结[MASK][MASK]来，让我又重新串是一遍那些思想"
gt = "收获颇丰。算是作者的同行，最近也在做类似的东西，但是作者都给总结了起来，让我又重新串了一遍那些思想"

prompt = '今天天空晴朗'
next_sentence = '算是作者的同行'


batch = tokenizer(
    prompt,
    next_sentence,
    add_special_tokens=True,
    padding='max_length',
    truncation='longest_first',
    max_length=150,
    return_tensors='pt',
    return_attention_mask = True,
    return_token_type_ids=True
)

# next_sentence = tokenizer(
#     next_sentence,
#     add_special_tokens=True,
#     padding='max_length',
#     truncation='longest_first',
#     max_length=150,
#     return_tensors='pt',
#     return_attention_mask = True
# )

# mlm_model.to('cuda')

# output = mlm_model(
#     input_ids=a['input_ids'].to('cuda'),
#     attention_mask=a['attention_mask'].to('cuda'),
#     labels=gt['input_ids'].to('cuda')
# )

output = nsp_model(
    **batch,
    labels=torch.LongTensor([1])
)


pass