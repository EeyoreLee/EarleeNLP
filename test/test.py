# -*- encoding: utf-8 -*-
'''
@create_time: 2021/06/05 16:50:07
@author: lichunyu
'''

import dataclasses
from dataclasses import dataclass
import logging
import os
import sys
from typing import Optional, Dict, Union, Any

from datasets import ClassLabel, load_dataset, load_metric, Dataset
from sklearn.model_selection import train_test_split
from transformers.file_utils import PaddingStrategy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    BertTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    HfArgumentParser,
    Trainer,
    set_seed,
    PreTrainedTokenizerFast,
    PreTrainedTokenizerBase
)

from transformers.trainer_utils import get_last_checkpoint, is_main_process, is_torch_cuda_available
from transformers.utils import check_min_version
import transformers

def normalize_bbox(bbox, width, height):
     return [
         int(1000 * (bbox[0] / width)),
         int(1000 * (bbox[1] / height)),
         int(1000 * (bbox[2] / width)),
         int(1000 * (bbox[3] / height)),
     ]

pth = '/Users/lichunyu/Desktop/EarleeNLP/data/wanli_classify_auged_droped_0219.pkl'
df = pd.read_pickle(pth)
df_train, df_eval = train_test_split(df, test_size=0.995, random_state=42)
train_dataset = Dataset.from_pandas(df_train)
eval_dataset = Dataset.from_pandas(df_eval)
pass

@dataclass
class DataCollatorForKeyValueExtraction:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        print(features)
        input_ids = []
        attention_masks = []
        labels = []
        bbox = []
        label_name = "label" if "label" in features[0].keys() else "target"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        has_image_input = "image" in features[0]
        has_bbox_input = "bbox" in features[0]

        for feat in features:
            token_boxes = []

            for w, box in zip(feat['text'], feat['coor']):
                word_tokens = self.tokenizer.tokenize(w)
                token_boxes.extend([normalize_bbox(box, feat['width'], feat['height'])] * len(word_tokens))

                if len(token_boxes) < self.max_length-2:
                    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]] + (self.max_length-2-len(token_boxes))*[[0,0,0,0]]
                else:
                    token_boxes = [[0, 0, 0, 0]] + token_boxes[:self.max_length-2] + [[1000, 1000, 1000, 1000]]

                batch = self.tokenizer(
                    ' '.join(feat['text']),
                    add_special_tokens = True,
                    truncation='longest_first',
                    max_length = self.max_length,
                    padding = 'max_length',
                    return_attention_mask = True,
                    return_tensors = 'pt',
                    )

                bbox.append(token_boxes)
                input_ids.append(batch['input_ids'])
                attention_masks.append(batch['attention_mask'])
    
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        bbox = torch.tensor(bbox)

        # return input_ids, attention_masks, labels, bbox
        return {
            'input_ids': input_ids,
            'attention_masks': attention_masks,
            'labels': labels,
            'bbox': bbox
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

datacollator = DataCollatorForKeyValueExtraction(
    tokenizer=tokenizer,
    max_length=510
)

res = datacollator(train_dataset[0:8])
pass