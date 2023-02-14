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
    AutoTokenizer
)

from .._base.base_collection import (
    BaseCollection
)
from .._base.base_collator import (
    MlmDataCollatorWithPadding
)


class Collection(BaseCollection):

    def __init__(self, preprocessing_num_workers=1, load_from_cache_file=False, *args, **kwds) -> None:
        super().__init__(*args, **kwds)
        self._init_tokenzier()
        self.preprocessing_num_workers = preprocessing_num_workers
        self.load_from_cache_file = load_from_cache_file

    def _init_tokenzier(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)

    def collect(self):
        dev_dataset = None  # Unsupervised learning, not available for dev dataset
        extension = self.train_data_path.split('.')[-1]
        if extension == "txt":
            extension = "text"
        if extension == "csv":
            datasets = load_dataset(extension, data_files=self.train_data_path, cache_dir="data/cache_dir", delimiter=",")
        else:
            datasets = load_dataset(extension, data_files=self.train_data_path, cache_dir="data/cache_dir")

        column_names = datasets["train"].column_names
        sent2_cname = None
        if len(column_names) == 2:
            # Pair datasets
            sent0_cname = column_names[0]
            sent1_cname = column_names[1]
        elif len(column_names) == 3:
            # Pair datasets with hard negatives
            sent0_cname = column_names[0]
            sent1_cname = column_names[1]
            sent2_cname = column_names[2]
        elif len(column_names) == 1:
            # Unsupervised datasets
            sent0_cname = column_names[0]
            sent1_cname = column_names[0]
        else:
            raise NotImplementedError

        def prepare_features(examples):
            # padding = longest (default)
            #   If no sentence in the batch exceed the max length, then use
            #   the max sentence length in the batch, otherwise use the 
            #   max sentence length in the argument and truncate those that
            #   exceed the max length.
            # padding = max_length (when pad_to_max_length, for pressure test)
            #   All sentences are padded/truncated to data_args.max_seq_length.
            total = len(examples[sent0_cname])

            # Avoid "None" fields 
            for idx in range(total):
                if examples[sent0_cname][idx] is None:
                    examples[sent0_cname][idx] = " "
                if examples[sent1_cname][idx] is None:
                    examples[sent1_cname][idx] = " "

            sentences = examples[sent0_cname] + examples[sent1_cname]

            # If hard negative exists
            if sent2_cname is not None:
                for idx in range(total):
                    if examples[sent2_cname][idx] is None:
                        examples[sent2_cname][idx] = " "
                sentences += examples[sent2_cname]

            sent_features = self.tokenizer(
                sentences,
                max_length=self.max_length,
                truncation=True,
                padding="max_length"
            )

            features = {}
            if sent2_cname is not None:
                for key in sent_features:
                    features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
            else:
                for key in sent_features:
                    features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
                
            return features

        train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            num_proc=self.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=self.load_from_cache_file,
        )
        # train_dataset.remove_columns([self.label_name])

        return train_dataset, train_dataset


def data_collator(**kwds):
    collator = MlmDataCollatorWithPadding(**kwds)
    return collator