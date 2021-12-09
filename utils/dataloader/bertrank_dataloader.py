# -*- encoding: utf-8 -*-
'''
@create_time: 2021/12/09 14:44:59
@author: lichunyu
'''


import torch
from torch.utils.data import Dataset


class RankDataset(Dataset):

    def __init__(self, df, tokenizer, max_length=128, higher_text_colname='higher', lower_text_colname='lower') -> None:
        super().__init__()
        self.higher_text_list = df[higher_text_colname].to_list()
        self.lower_text_list = df[lower_text_colname].to_list()
        assert len(self.higher_text_list) == len(self.lower_text_list)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.higher_text_list)

    def __getitem__(self, index: int):
        higher_encode = self.tokenizer(
            self.higher_text_list[index],
            truncation='longest_first',
            max_length = self.max_length,
            padding = 'max_length',
            return_attention_mask = True,
            return_tensors = 'pt',
        )
        input_ids = higher_encode['input_ids'][0]
        attention_mask = higher_encode['attention_mask'][0]
        lower_encode = self.tokenizer(
            self.lower_text_list[index],
            truncation='longest_first',
            max_length = self.max_length,
            padding = 'max_length',
            return_attention_mask = True,
            return_tensors = 'pt',
        )
        lower_input_ids = lower_encode['input_ids'][0]
        lower_attention_mask = lower_encode['attention_mask'][0]
        target = torch.tensor(1)
        return input_ids, attention_mask, lower_input_ids, lower_attention_mask, target