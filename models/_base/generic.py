# -*- encoding: utf-8 -*-
'''
@create_time: 2022/08/16 18:33:21
@author: lichunyu
'''

_default_tokenizer_param = {
    "add_special_tokens": True,
    "truncation": "longest_first",
    "max_length": 150,
    "padding": "max_length",
    "return_attention_mask": True,
    "return_tensors": "pt"
}