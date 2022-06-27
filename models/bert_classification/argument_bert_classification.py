# -*- encoding: utf-8 -*-
'''
@create_time: 2022/06/24 10:45:09
@author: lichunyu
'''
from dataclasses import dataclass, field


@dataclass
class ModelArgument:

    author: str = field(default="EeyoreLee", metadata={"help": "author"})
    init_param: dict = field(default={}, metadata={"help": "param for model's initialization"})