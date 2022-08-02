# -*- encoding: utf-8 -*-
'''
@create_time: 2022/06/27 11:49:55
@author: lichunyu
'''
import torch
from torch.utils.data import Dataset
import pandas as pd


class BaseCollection(object):

    def __init__(
        self,
        data_path=None,
        train_data_path=None,
        dev_data_path=None,
        tokenizer_name_or_path=None,
        test_size=0.2,
        label_name="label",
        data_name="review",  # TODO Temporary test use
        **kwargs
    ) -> None:
        self.uncut = True if train_data_path is None else False
        self.data_path = data_path
        self.train_data_path = train_data_path
        self.dev_data_path = dev_data_path
        self.test_size = test_size
        self.label_name = label_name
        self.data_name = data_name
        self.tokenizer_name_or_path = tokenizer_name_or_path


class ClassificationDataset(Dataset):

    def __init__(self, data, tokenizer, tokenizer_param=None, label_name="label", data_name="text") -> None:
        self.raw_data = data
        self.label_name = label_name
        self.data_name = data_name
        self.tokenizer = tokenizer
        default_param = {
            "add_special_tokens": True,
            "truncation": "longest_first",
            "max_length": 150,
            "padding": "max_length",
            "return_attention_mask": True,
            "return_tensors": "pt"
        }
        self.tokenizer_param = default_param if tokenizer_param is None else tokenizer_param
        self.data = self._tokenizer_all()

    def _tokenizer_all(self):
        tokenizered_data = []
        for idx, row in self.raw_data.iterrows():
            batch = self.tokenizer(
                row[self.data_name],
                **self.tokenizer_param
            )
            batch = {k: v[0] for k, v in batch.items()}
            batch["labels"] = torch.tensor(row[self.label_name])
            tokenizered_data.append(batch)
        return tokenizered_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def classification_collate_fn(batch_dict, data_name, label_name):
    ...