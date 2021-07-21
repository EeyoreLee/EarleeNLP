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

device = 'cuda'

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
    input_ids = batch.input_ids
    attention_mask = batch.attention_mask
    token_type_ids = batch.token_type_ids
    return input_ids, attention_mask, token_type_ids



m = torch.load('/ai/223/person/lichunyu/models/ner-mrc/bert-2021-07-19-14-20-44-f1_86.pth')
model = m.module

input_ids, attention_mask, token_type_ids = encode(
    "按照地理位置划分的国家,城市,乡镇,大洲",
    "我们藏有一册１９４５年６月油印的《北京文物保存保管状态之调查报告》，调查范围涉及故宫、历博、古研所、北大清华图书馆、北图、日伪资料库等二十几家，言及文物二十万件以上，洋洋三万余言，是珍贵的北京史料。"
)
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



pass
