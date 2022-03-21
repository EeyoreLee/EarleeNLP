# -*- encoding: utf-8 -*-
'''
@create_time: 2022/03/14 17:58:54
@author: lichunyu
'''
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence, pad_sequence
import torch
from torch.utils.data import DataLoader, Dataset

from transformers import BertModel, BertTokenizer, BertConfig



class BertRnnClassification(nn.Module):

    def __init__(self, bert_config_or_path, rnn_hidden_size, num_class=10, need_pack=False):
        super().__init__()
        self.need_pack = need_pack
        self.bert_config = BertConfig.from_pretrained(bert_config_or_path)
        self.bert = BertModel.from_pretrained(bert_config_or_path)
        self.rnn = nn.RNN(input_size=self.bert_config.hidden_size, hidden_size=rnn_hidden_size, batch_first=True)
        self.cls = nn.Linear(rnn_hidden_size, num_class)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = bert_output[0]
        if self.need_pack is False:
            output = self.rnn(sequence_output)
            output = output[0]
        classification = self.cls(output)
        return classification



class BertRnnDataset(Dataset):
    ...



if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('/ai/223/person/lichunyu/pretrain-models/bert-base-chinese')
    model = BertRnnClassification(
        bert_config_or_path='/ai/223/person/lichunyu/pretrain-models/bert-base-chinese',
        rnn_hidden_size=512
    )
    text = '今天是星期三'
    inputs = tokenizer(
        text=text,
        truncation='longest_first',
        add_special_tokens=True,
        max_length = 20,
        padding = 'max_length',
        return_attention_mask = True,
        return_tensors = 'pt',
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']
    output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    ...