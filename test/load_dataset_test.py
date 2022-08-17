from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Any, Union
import sys
sys.path.append(".")

from transformers import (
    PreTrainedTokenizerBase,
    AutoTokenizer,
    BertTokenizer
)
import torch
from models._base.generic import _default_tokenizer_param


dataset = load_dataset(
    "json",
    data_files={
        "train": "/disk/223/person/lichunyu/datasets/public_data/clue/tnews_public/train.json",
        "dev": "/disk/223/person/lichunyu/datasets/public_data/clue/tnews_public/dev.json"
    }
)

@dataclass
class BaseTokenization:

    tokenizer: Union[str, PreTrainedTokenizerBase] = field(default=None)
    tokenizer_param: dict = field(default_factory=dict)
    data_name: str = field(default="sentence")
    label_name: str = field(default="label")

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
        label = examples[self.label_name]
        if label and not isinstance(label[0], int):
            label = [int(_) for _ in label]
        batch[self.label_name] = label
        return batch

prepare_features = BaseTokenization(tokenizer="/disk/223/person/lichunyu/pretrain-models/bert-base-chinese")

dataset = dataset.map(
    prepare_features,
    batched=True
)

...