# -*- encoding: utf-8 -*-
'''
@create_time: 2022/06/24 10:55:36
@author: lichunyu
'''
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)

from .._base.base_collection import (
    BaseCollection,
    BaseTokenization
)
from .._base.base_collator import (
    MlmDataCollatorWithPadding,
)


class Collection(BaseCollection):

    def __init__(self, load_from_cache_file=False, *args, **kwds) -> None:
        super().__init__(*args, **kwds)
        self._init_tokenzier()
        self.load_from_cache_file = load_from_cache_file

    def _init_tokenzier(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)

    def collect(self):
        data_files = {
            "train": self.data_path if self.uncut else self.train_data_path
        }
        if self.dev_data_path is not None:
            data_files["dev"] = self.dev_data_path
        sample = data_files["train"]
        if isinstance(sample, list):
            sample = sample[0]
        extension = sample.split('.')[-1]
        if extension == "txt":
            extension = "text"
        if extension == "tsv":
            dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
        else:
            dataset = load_dataset(extension, data_files=data_files)
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
        dataset = dataset.remove_columns([self.label_name])
        if self.dev_data_path is not None:
            train_dataset = dataset["train"]
            dev_dataset = dataset["dev"]
        else:
            train_dataset = dataset["train"]
            dev_dataset = None
        return train_dataset, dev_dataset


def data_collator(model_name_or_path, mlm=True, mlm_probability=0.15, **kwds):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # tokenizer.pad_token = tokenizer.eos_token
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
    return collator