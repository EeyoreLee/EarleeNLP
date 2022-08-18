# -*- encoding: utf-8 -*-
'''
@create_time: 2022/08/17 15:55:04
@author: lichunyu
'''
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class BaseModelArgument:

    author: str = field(default="EeyoreLee", metadata={"help": "author"})
    model_init_param: dict = field(default_factory=dict, metadata={"help": "param for model's initialization"})
    collection_param: dict = field(default_factory=dict, metadata={"help": "param for collection's initialization"})
    collator_param: dict = field(default_factory=dict, metadata={"help": "param for DataCollator initialization"})