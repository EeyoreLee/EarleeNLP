# -*- encoding: utf-8 -*-
'''
@create_time: 2022/06/24 10:55:36
@author: lichunyu
'''
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer
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
        self.tokenizer = BertTokenizer.from_pretrained("/disk/223/person/lichunyu/pretrain-models/bert-base-chinese")

    def collect(self):
        dataloader_train, dataloader_dev = self._collect_by_csv()
        ...

    def _collect_by_csv(self):
        if self.uncut is True:
            df = pd.read_csv(self.data_path)
            df_train, df_dev = train_test_split(
                df,
                test_size=self.test_size,
                stratify=df[self.label_name]
            )
        else:
            df_train = pd.read_csv(self.train_data_path)
            df_dev = pd.read_csv(self.dev_data_path)

        dataset_train = ClassificationDataset(
            df_train,
            tokenizer=self.tokenizer,
            label_name=self.label_name,
            data_name=self.data_name
        )
        dataset_dev = ClassificationDataset(
            df_dev,
            label_name=self.label_name,
            data_name=self.data_name
        )
        dataloader_train = DataLoader(
            dataset=dataset_train,
            batch_size=self.batch_size
        )
        ...





def collate_fn(batch_dict, data_name, label_name):
    ...