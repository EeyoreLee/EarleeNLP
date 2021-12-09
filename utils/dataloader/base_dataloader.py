# -*- encoding: utf-8 -*-
'''
@create_time: 2021/12/09 15:06:24
@author: lichunyu
'''


import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self, df, tokenizer, max_length, text_colname='text') -> None:
        super().__init__()
        self.text_list = df[text_colname].to_list()
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