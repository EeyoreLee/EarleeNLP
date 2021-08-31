# -*- encoding: utf-8 -*-
'''
@create_time: 2021/08/31 14:38:27
@author: lichunyu
'''

import torch
from transformers import BertForTokenClassification, BertConfig, BertTokenizer



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1

outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits
pass