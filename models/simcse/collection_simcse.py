# -*- encoding: utf-8 -*-
'''
@create_time: 2022/06/24 10:55:36
@author: lichunyu
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    AutoTokenizer
)

from .._base.base_collection import (
    BaseCollection,
    ClassificationDataset
)


class Collection(BaseCollection):

    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)
        self._init_tokenzier()

    def _init_tokenzier(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)

    def collect(self):
        train_dataset, dev_dataset = self._collect_by_txt()
        return train_dataset, dev_dataset

    def _collect_by_txt(self):
        with open(self.data_path, "r") as f:
            data = f.read().splitlines()
        