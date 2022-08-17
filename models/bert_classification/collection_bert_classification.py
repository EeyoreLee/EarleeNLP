# -*- encoding: utf-8 -*-
'''
@create_time: 2022/06/24 10:55:36
@author: lichunyu
'''
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    BertTokenizer
)
from datasets import load_dataset

from .._base.base_collection import (
    BaseCollection,
    ClassificationDataset,
    BaseTokenization
)
from .._base.base_collator import (
    BaseDataCollator
)


class Collection(BaseCollection):

    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)
        self._init_tokenzier()

    def _init_tokenzier(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)

    def collect(self):
        if self.uncut:
            extension = self.data_path.split(".")[-1]
        else:
            extension = self.train_data_path.split(".")[-1]  # TODO Support different extension between train and dev dataset
        if extension == "csv":
            train_dataset, dev_dataset = self._collect_by_csv()
        elif extension == "tsv":
            train_dataset, dev_dataset = self._collect_by_tsv()
        elif extension == "json":
            train_dataset, dev_dataset = self._collect_by_json()
        else:
            raise NotImplementedError
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

    def _collect_by_json(self):
        data_files = {
            "train": self.data_path if self.uncut else self.train_data_path
        }
        if self.dev_data_path is not None:
            data_files["dev"] = self.dev_data_path
        dataset = load_dataset("json", data_files=data_files)
        tokenization = BaseTokenization(
            tokenizer=self.tokenizer,
            data_name=self.data_name,
            label_name=self.label_name,
            label_map=self.label_map
        )
        dataset = dataset.map(
            tokenization,
            batched=True,
            num_proc=self.num_proc
        )
        if self.dev_data_path is not None:
            train_dataset = dataset["train"]
            dev_dataset = dataset["dev"]
        else:
            ...
        return train_dataset, dev_dataset


DataCollator = BaseDataCollator