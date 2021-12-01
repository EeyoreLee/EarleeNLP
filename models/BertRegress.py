# -*- encoding: utf-8 -*-
'''
@create_time: 2021/12/01 10:51:09
@author: lichunyu
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertModel, BertConfig


class BertRegress(nn.Module):

    def __init__(self):
        super().__init__()
        self.config = BertConfig.from_pretrained('/ai/223/person/lichunyu/pretrain-models/bert-large-uncased')
        self.bert = BertModel.from_pretrained('/ai/223/person/lichunyu/pretrain-models/bert-large-uncased')
        self.regress = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, y):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        output = self.regress(pooled_output)
        if self.training:
            loss_func = nn.MSELoss()
            loss = loss_func(output, y)
            return {
                'loss': loss,
                'output': output
            }
        return {'output': output}


if __name__ == '__main__':
    model = BertRegress()
    pass