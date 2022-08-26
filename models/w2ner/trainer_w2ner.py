# -*- encoding: utf-8 -*-
'''
@create_time: 2022/07/29 16:17:13
@author: lichunyu
'''
import sys

from transformers.modeling_utils import unwrap_model
import torch.nn as nn

from .._base.base_trainer import BaseTrainer


class ModelTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        grid_labels = labels
        grid_mask2d = inputs["grid_mask2d"]

        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs[grid_mask2d], grid_labels[grid_mask2d])

        return (loss, outputs) if return_outputs else loss