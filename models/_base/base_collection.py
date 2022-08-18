# -*- encoding: utf-8 -*-
'''
@create_time: 2022/06/27 11:49:55
@author: lichunyu
'''
from dataclasses import dataclass, field
from typing import Union, Any

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import (
    PreTrainedTokenizerBase,
    AutoTokenizer
)

from .generic import _default_tokenizer_param
from utils.generic import MirrorDict


class BaseCollection(object):

    def __init__(
        self,
        data_path=None,
        train_data_path=None,
        dev_data_path=None,
        tokenizer_name_or_path=None,
        test_size=0.2,
        label_name="label",
        data_name="sentence",
        label_map=None,
        max_length=150,
        tokenizer_param=None,
        num_proc=4,
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
        self.max_length = max_length
        self.num_proc = num_proc
        self.tokenizer_param = _default_tokenizer_param if tokenizer_param is None else tokenizer_param
        self.label_map = MirrorDict() if label_map is None else label_map

        if isinstance(self.label_map, dict):
            self.label_map = {int(k): int(v) for k, v in self.label_map.items()}


class ClassificationDataset(Dataset):

    def __init__(self, data, tokenizer, tokenizer_param=None, label_name="label", data_name="text") -> None:
        self.raw_data = data
        self.label_name = label_name
        self.data_name = data_name
        self.tokenizer = tokenizer
        self.tokenizer_param = _default_tokenizer_param if tokenizer_param is None else tokenizer_param
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


@dataclass
class BaseTokenization:

    tokenizer: Union[str, PreTrainedTokenizerBase] = field(default=None)
    tokenizer_param: dict = field(default_factory=dict)
    data_name: str = field(default="sentence")
    label_name: str = field(default="label")
    label_map: dict = field(default_factory=MirrorDict)

    def __post_init__(self):
        if isinstance(self.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        if not self.tokenizer_param:
            self.tokenizer_param = _default_tokenizer_param
        if "return_tensors" in self.tokenizer_param:
            del self.tokenizer_param["return_tensors"]

    def __call__(self, examples) -> Any:
        sentence = examples[self.data_name]
        batch = self.tokenizer(
            sentence,
            **self.tokenizer_param
        )
        if self.label_name in examples:
            label = examples[self.label_name]
            if label and not isinstance(label[0], int):
                label = [self.label_map[int(_)] for _ in label]
            else:
                label = [self.label_map[_] for _ in label]
            batch[self.label_name] = label
        return batch
