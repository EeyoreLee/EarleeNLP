# -*- encoding: utf-8 -*-
'''
@create_time: 2021/07/02 17:09:41
@author: lichunyu
'''


from transformers import XLNetModel, XLNetTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F


tokenizer = XLNetTokenizer.from_pretrained('/Users/lichunyu/data/models/chinese-xlnet')
pass