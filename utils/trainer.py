# -*- encoding: utf-8 -*-
'''
@create_time: 2022/05/12 18:19:11
@author: lichunyu
'''
import logging
from packaging import version
import os
from datetime import datetime

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.metrics import f1_score, accuracy_score, classification_report, cohen_kappa_score
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

from .generic import get_args, setup_logger


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
        optimizer = None,
        strategy=None
    ) -> None:
        if args is None:
            output_dir = "tmp_output"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)
        self.args = args
        self.extra_args = extra_args
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
        self.output_dir = get_args("output_dir", args, extra_args, "tmp_output")
        self.deepspeed = None
        self.training = False
        self.model = model  # TODO place this line at the end of __init__() when development is complete.
        self.rank = -1
        self.f1_score = 0.0
        self.warped_model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.dev_unused_column: list = get_args("dev_unused_column", args, extra_args, [])
        self.train_unused_column: list = get_args("train_unused_column", args, extra_args, [])
        self.strategy = get_args("strategy", args, extra_args, None)
        self.log_file_path = get_args("log_file_path", args, extra_args, None)
        setup_logger(self.log_file_path)
        # args._setup_devices
        self.num_train_epochs = get_args("num_train_epochs", self.args, self.extra_args, 4)
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
        if self.args.world_size <= 1 and _is_torch_generator_available:  # BUG fix world_size
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
        if self.strategy is None:  # TODO use an Enum to instead of
            if _is_torch_generator_available:
                return RandomSampler(self.train_dataset, generator=generator)
            return RandomSampler(self.train_dataset)
        elif self.strategy == "ddp":
            return DistributedSampler(
                self.train_dataset,
                num_replicas=torch.cuda.device_count(),
                rank=self.rank,
                seed=seed
            )
        else:
            raise NotImplementedError(
                f"strategy {self.strategy} is not implemented. please choose in"
                f"null ddp"  # TODO get the strategy list
            )

    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        train_sampler = self._get_train_sampler()

        # TODO collate_fn  drop_last  num_workers pin_memory
        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            worker_init_fn=seed_worker,
        )

    def _get_dev_sampler(self):
        dev_dataset = self.dev_dataset
        # if self.args.world_size <= 1:  # BUG
        if self.strategy is None:
            return SequentialSampler(dev_dataset)
        elif self.strategy == "ddp":
            return ShardSampler(
                dev_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                num_processes=torch.cuda.device_count(),
                process_index=self.rank,
            )

    def get_dev_dataloader(self):
        dev_dataset = self.dev_dataset
        dev_sampler = self._get_dev_sampler()

        return DataLoader(
            dev_dataset,
            sampler=dev_sampler,
            batch_size=self.args.eval_batch_size,
            drop_last=self.args.dataloader_drop_last,
        )


    def fit(self, resume_from_checkpoint=None, **kwargs):
        self.training = True
        # TODO resume model   # line 1299 in transformers.trainer
        self._train_batch_size = get_args("train_batch_size", self.args, self.extra_args, 128)
        if self.strategy == "ddp":
            os.environ['MASTER_ADDR'] = "localhost"
            os.environ['MASTER_PORT'] = "12355"
            world_size = torch.cuda.device_count()
            mp.spawn(
                self.inner_training_loop,
                args=(world_size,),
                nprocs=world_size,
                join=True
            )
        else:
            return self.inner_training_loop(
                resume_from_checkpoint=resume_from_checkpoint,
                **kwargs
            )

    def inner_training_loop(self, rank=-1, world_size=1, resume_from_checkpoint=None, **kwargs):
        self.rank = rank
        if self.strategy == "ddp":
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)
        logger.setLevel(logging.INFO if rank in [-1, 0] else logging.WARNING)
        logger.info(f" train batch size is : {self._train_batch_size}")
        train_dataloader = self.get_train_dataloader()
        dev_dataloader = self.get_dev_dataloader()
        model = self.model.cuda()
        model = self._warp_model(model)
        if model is not self.warped_model:
            self.warped_model = model
        optimizer = self.create_optimizer()
        model.train()
        for epoch in range(self.num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            logger.info(f" training  epoch:{epoch}/{self.num_train_epochs}")
            total_train_loss = 0.0
            for step, batch in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                batch = {k: v.cuda() for k, v in batch.items()}
                model.zero_grad()
                output = model(**batch)
                loss = output.loss
                loss = loss.mean()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            logger.info(f"Average training loss: {total_train_loss/len(train_dataloader)}")

            logger.info(f" evaling ...")  # TODO Compatible with distributed evaluation
            model.eval()
            total_dev_preds = []
            total_dev_labels = []
            for step, batch in tqdm.tqdm(enumerate(dev_dataloader), total=len(dev_dataloader)):
                labels = batch["labels"]  # TODO dynamic label name
                batch = {k: v.cuda() for k, v in batch.items() if k not in self.dev_unused_column}
                with torch.no_grad():
                    output = model(**batch)
                    loss = output.loss
                    loss = loss.mean()
                    logits = output.logits

                total_dev_preds.extend(np.argmax(logits.detach().cpu().numpy(), axis=-1).flatten().tolist())
                total_dev_labels.extend(labels.cpu().numpy().flatten().tolist())

            f1 = f1_score(total_dev_labels, total_dev_preds, average="micro")
            logger.info(f" f1 score for dev dataset is : {f1}")
            logger.info(f'\n{classification_report(total_dev_preds, total_dev_labels, zero_division=1)}')

            if rank in [-1, 0]:
                if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
                    torch.save(model.module, f"{self.output_dir}/model-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-f1-{str(int(f1*100))}.pth")
                else:
                    torch.save(model, f"{self.output_dir}/model-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-f1-{str(int(f1*100))}.pth")
        ...

    def _warp_model(self, model):
        if False:  # TODO fp16 deepspeed and so on
            return model
        # elif self.args.local_rank != -1:  # BUG
        elif self.strategy=="ddp":
            kwargs = {}
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids = [self.rank],
                # output_device= self.args.local_rank if self.args._n_gpu != 0 else None,
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

    def compute_f1(self, preds, labels):
        flat_preds = np.argmax(preds, axis=1).flatten()
        flat_labels = labels.flatten()
        f1 = f1_score(flat_labels, flat_preds, average="micro")
        return f1
