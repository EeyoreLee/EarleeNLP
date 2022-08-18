# -*- encoding: utf-8 -*-
'''
@create_time: 2022/07/29 16:17:13
@author: lichunyu
'''
from typing import Dict, List, Optional

import torch
from torch.utils.data.dataset import Dataset

from .._base.base_trainer import BaseTrainer


class ModelTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
