# -*- encoding: utf-8 -*-
'''
@create_time: 2021/02/17 14:04:13
@author: lichunyu
'''

import torch
import torch.nn as nn
from transformers import (
    BertTokenizer
)

import os


def test():

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'

    device = 'cuda'


    class DFTest(object):

        def __init__(self):
            ...


    tokenizer = BertTokenizer.from_pretrained('/root/pretrain-models/bert-base-chinese')
    def encode(query, context):
        batch = tokenizer(
            query,
            context,
            add_special_tokens=True,
            truncation='longest_first',
            max_length=150,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        return batch
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        token_type_ids = batch.token_type_ids
        return input_ids, attention_mask, token_type_ids



    # m = torch.load('/ai/223/person/lichunyu/models/ner-mrc/bert-2021-07-19-14-20-44-f1_86.pth')
    model = torch.load('/ai/223/person/lichunyu/models/tmp/bert-2021-07-28-15-18-42-f1_64.pth')
    # model = m.module

    input_ids = []
    attention_mask = []
    token_type_ids = []

    query_list = ['发生某一事情的确定的日子或时期', '人口集中，居民以非农业人口为主，工商业比较发达的地区', '工作时所需用的器具']
    text = '****'

    for query in query_list:
        batch = encode(
            query,
            text
        )

        input_ids.append(batch['input_ids'])
        attention_mask.append(batch['attention_mask'])
        token_type_ids.append(batch['token_type_ids'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)

    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    token_type_ids = token_type_ids.cuda()

    model.eval()

    with torch.no_grad():
        start_logits, end_logits, span_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )


    def compute_ner(text, _start_logits, _end_logits, _span_logits, _token_type_ids):
        if len(_start_logits.shape) == 1:
            _start_logits = _start_logits[None, :]
        if len(_end_logits.shape) == 1:
            _end_logits = _end_logits[None, :]
        if len(_span_logits.shape) == 2:
            _span_logits = _span_logits[None, :, :]
        if len(_token_type_ids.shape) == 1:
            _token_type_ids = _token_type_ids[None, :]

        res = []
        text_token_idx = torch.where(_token_type_ids == 1)[-1]
        start_text_token_idx = text_token_idx[0]
        end_text_token_idx = text_token_idx[-1]
        start_tag = torch.where(_start_logits[:,start_text_token_idx:end_text_token_idx+1] > 0)[-1]
        end_tag = torch.where(_end_logits[:,start_text_token_idx:end_text_token_idx+1] > 0)[-1]
        span = span_logits[0][start_text_token_idx:end_text_token_idx+1][:,start_text_token_idx:end_text_token_idx+1]
        for t in start_tag:
            end = torch.argmax(span[t,end_tag])
            text_end_idx = end_tag[end]
            if text_end_idx < end:
                continue
            res.append(text[t:text_end_idx+1])
        # print(res)
        return res


    for i in range(len(query_list)):
        _ = compute_ner(
            text,
            start_logits[i],
            end_logits[i],
            span_logits[i],
            token_type_ids[i]
        )

    return 0
