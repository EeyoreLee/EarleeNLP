# -*- encoding: utf-8 -*-
'''
@create_time: 2022/06/24 10:55:36
@author: lichunyu
'''
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
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name_or_path)

    def collect(self):
        # train_dataset, dev_dataset = self._collect_by_csv()
        train_dataset, dev_dataset = self._collect_by_tsv()
        return train_dataset, dev_dataset

    def _collect_by_csv(self, delimiter=None):
        if self.uncut is True:
            df = pd.read_csv(self.data_path, delimiter=delimiter)
            # df, _ = train_test_split(
            #     df,
            #     test_size=0.99,
            #     stratify=df[self.label_name]
            # )
            df_train, df_dev = train_test_split(
                df,
                test_size=self.test_size,
                stratify=df[self.label_name]
            )
        else:
            df_train = pd.read_csv(self.train_data_path, delimiter=delimiter)
            df_dev = pd.read_csv(self.dev_data_path, delimiter=delimiter)

        train_dataset = ClassificationDataset(
            df_train,
            tokenizer=self.tokenizer,
            label_name=self.label_name,
            data_name=self.data_name
        )
        dev_dataset = ClassificationDataset(
            df_dev,
            tokenizer=self.tokenizer,
            label_name=self.label_name,
            data_name=self.data_name
        )
        return train_dataset, dev_dataset

    def _collect_by_tsv(self):
        return self._collect_by_csv(delimiter='\t')

