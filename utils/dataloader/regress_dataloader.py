# -*- encoding: utf-8 -*-
'''
@create_time: 2021/12/01 14:35:33
@author: lichunyu
'''

import torch
from torch.utils.data import Dataset


class JigsawDataset(Dataset):

    def __init__(self, df, tokenizer, max_length=300) -> None:
        super().__init__()
        self.ys = df['y'].to_list()
        self.text_list = df['comment_text'].to_list()
        assert len(self.ys) == len(self.text_list)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.text_list)

    def __getitem__(self, index: int):
        encode = self.tokenizer(
            self.text_list[index],
            truncation='longest_first',
            max_length = self.max_length,
            padding = 'max_length',
            return_attention_mask = True,
            return_tensors = 'pt',
        )
        input_ids = encode['input_ids'][0]
        attention_mask = encode['attention_mask'][0]
        y = torch.tensor(self.ys[index])
        return input_ids, attention_mask, y


class JigsawValDataset(Dataset):

    def __init__(self, df, text_name, tokenizer, max_length=128) -> None:
        super().__init__()
        self.text_list = df[text_name].to_list()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.text_list)

    def __getitem__(self, index: int):
        encode = self.tokenizer(
            self.text_list[index],
            truncation='longest_first',
            max_length = self.max_length,
            padding = 'max_length',
            return_attention_mask = True,
            return_tensors = 'pt',
        )
        input_ids = encode['input_ids'][0]
        attention_mask = encode['attention_mask'][0]
        return input_ids, attention_mask