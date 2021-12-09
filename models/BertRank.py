# -*- encoding: utf-8 -*-
'''
@create_time: 2021/12/09 11:03:57
@author: lichunyu
'''
from transformers import AutoModel, AutoConfig, BertConfig
import torch.nn as nn


class BertRank(nn.Module):

    def __init__(self, model_name_or_path='bert-base-uncased', config_name_or_path=None, drop_prob=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.config = AutoConfig.from_pretrained(config_name_or_path if config_name_or_path is not None else model_name_or_path)
        self.ln = nn.LayerNorm(self.config.hidden_size)
        self.dropout = nn.Dropout(drop_prob)
        self.linear = nn.Linear(self.config.hidden_size, 1)
        self.loss_fn = nn.MarginRankingLoss()

    def forward(self, input_ids, attention_mask, lower_input_ids=None, lower_attention_mask=None, target=None):
        # higher rank calculate
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        pooled_output = self.ln(pooled_output)
        pooled_output = self.dropout(pooled_output)
        metric = self.linear(pooled_output)

        result = {'metric': metric}

        if self.training:
            # lower rank calculate
            lower_pooled_output = self.bert(input_ids=lower_input_ids, attention_mask=lower_attention_mask)[1]
            lower_pooled_output = self.ln(lower_pooled_output)
            lower_pooled_output = self.dropout(lower_pooled_output)
            lower_metric = self.linear(lower_pooled_output)

            loss = self.loss_fn(metric, lower_metric, target)

            result['lower_metric'] = lower_metric
            result['loss'] = loss

        return result
