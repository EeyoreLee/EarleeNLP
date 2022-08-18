# -*- encoding: utf-8 -*-
'''
@create_time: 2022/08/12 10:44:05
@author: lichunyu
'''
from dataclasses import dataclass, field
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import (
    BertForMaskedLM,
    BertConfig
)


FurtherBert = BertForMaskedLM


def model_init(model: BertForMaskedLM, model_name_or_path, **kwds):
    config = BertConfig.from_pretrained(model_name_or_path, is_decoder=False)
    model = model.from_pretrained(model_name_or_path, config=config)
    return model