# -*- encoding: utf-8 -*-
'''
@create_time: 2022/05/12 18:19:11
@author: lichunyu
'''
import logging
from packaging import version
import os

import tqdm
import torch
import torch.nn as nn
from transformers import Trainer as Trainer_HF
from transformers import (
    TrainingArguments,
    enable_full_determinism,
    set_seed
)
from transformers.utils.generic import (
    cached_property
)
from transformers.trainer_utils import (
    seed_worker
)
from transformers.trainer_pt_utils import (
    ShardSampler,
    get_parameter_names
)
from torch.utils.data import (
    RandomSampler,
    DistributedSampler,
    DataLoader,
    SequentialSampler
)

from .generic import get_args


if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
else:
    _is_torch_generator_available = False


logger = logging.getLogger(__name__)


class Trainer():

    def __init__(
        self,
        model = None,
        args = None,
        extra_args: list = None,
        train_dataset = None,
        dev_dataset = None,
        callbacks = None,
        optimizer = None
    ) -> None:
        if args is None:
            output_dir = "tmp_output"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)
        self.args = args
        self.extra_args = extra_args
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
        self.deepspeed = None
        self.training = False
        self.model = model  # TODO place this line at the end of __init__() when development is complete.
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        args._setup_devices
        self.epoch = get_args("epoch", self.args, self.extra_args, 4)
        self.place_model_on_device = get_args("place_model_on_device", self.args, self.extra_args, False)
        self.is_model_parallel = False
        if hasattr(model, "is_parallelizable") and model.is_parallelizable and model.model_parallel:
            self.is_model_parallel = True
        if (self.is_model_parallel or self.deepspeed):
            self.place_model_on_device = False
        if self.place_model_on_device:
            self._move_model_to_device(model, get_args("device", self.args, self.extra_args, torch.device("cuda:0")))
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        # TODO Label smoothing # line 534 in transformers.trainer

    def _move_model_to_device(self, model, device):
        model = model.to(device)

    def _get_train_sampler(self):
        if self.train_dataset is None:
            return None
        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        # if self.args.world_size <= 1:  # BUG
        if False:
            if _is_torch_generator_available:
                return RandomSampler(self.train_dataset, generator=generator)
            return RandomSampler(self.train_dataset)
        else:
            return DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=seed
            )

    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            # collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def _get_dev_sampler(self):
        dev_dataset = self.dev_dataset
        if self.args.world_size <= 1:
            return SequentialSampler(dev_dataset)
        else:
            return ShardSampler(
                dev_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                num_processes=self.args.world_size,
                process_index=self.args.process_index,
            )

    def get_dev_dataloader(self):
        dev_dataset = self.dev_dataset
        dev_sampler = self._get_dev_sampler()

        return DataLoader(
            dev_dataset,
            sampler=dev_sampler,
            batch_size=self.args.eval_batch_size,
            # collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


    def fit(self, resume_from_checkpoint=None, **kwargs):
        self.training = True
        # TODO resume model   # line 1299 in transformers.trainer
        return self.inner_training_loop(
            resume_from_checkpoint=resume_from_checkpoint,
            **kwargs
        )

    def inner_training_loop(self, resume_from_checkpoint=None, **kwargs):
        self._train_batch_size = get_args("train_batch_size", self.args, self.extra_args, 32)
        train_dataloader = self.get_train_dataloader()
        dev_dataloader = self.get_dev_dataloader()
        model = self._warp_model(self.model)
        optimizer = self.create_optimizer()
        data_kwargs = dict(device=self.args.device)
        model.train()
        for epoch in range(self.epoch):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            for batch in tqdm.tqdm(train_dataloader):
                batch = {k: v.to(**data_kwargs) for k, v in batch.items()}
                output = model(**batch)
                ...

    def _warp_model(self, model):
        if False:  # TODO fp16 deepspeed and so on
            return model
        # elif self.args.local_rank != -1:  # BUG
        else:
            kwargs = {}
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids = [self.args.local_rank] if self.args._n_gpu != 0 else None,
                output_device= self.args.local_rank if self.args._n_gpu != 0 else None,
                **kwargs
            )
        return model

    def create_optimizer(self):
        model = self.model
        if self.optimizer is None:
            decay_parameters = get_parameter_names(model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer_HF.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

