# -*- encoding: utf-8 -*-
'''
@create_time: 2022/05/12 18:19:11
@author: lichunyu
'''
import logging

import torch
from transformers import (
    TrainingArguments,
    enable_full_determinism,
    set_seed
)
from transformers.utils.generic import (
    cached_property
)


logger = logging.getLogger(__name__)


class Trainer():

    def __init__(
        self,
        model = None,
        args = None,
        extra_args: list = None,
        train_dataloader = None,
        dev_dataloader = None,
        callbacks = None,
        optimizers = None
    ) -> None:
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
        self.deepspeed = None
        self.training = False

    def fit():
        ...


if __name__ == "__main__":
    # enable_full_determinism(42)
    # torch.use_deterministic_algorithms(True)
    ...
