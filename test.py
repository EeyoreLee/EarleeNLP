# -*- encoding: utf-8 -*-
'''
@create_time: 2021/10/26 18:27:23
@author: lichunyu
'''
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('/ai/223/person/lichunyu/pretrain-models/bert-base-chinese')
res = tokenizer('身份卡好几十块话费卡', return_tensors='pt')
pass