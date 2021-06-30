# -*- encoding: utf-8 -*-
'''
@create_time: 2021/06/24 11:08:01
@author: lichunyu
'''

from importlib.metadata import metadata
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from transformers import (
    BertModel,
    BertTokenizerFast,
    BertConfig,
    TrainingArguments,
    HfArgumentParser,
    Trainer,
    set_seed,
    PreTrainedTokenizerBase,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
import transformers
from datasets import Dataset, load_dataset
from enum import Enum

import dataclasses
from dataclasses import dataclass
import logging
import os
import sys
from typing import Union, Optional, Any


check_min_version('4.7.0')


class SpecialToken(Enum):

    SIMSEP = '<SIMSEP>'


class SBert(nn.Module):
    """[Sentence-Bert]

    :param nn: [description]
    :type nn: [type]
    """    

    def __init__(self, bert_pretrained_name='bert-base-chinese', pool_stride=1, cosin_eps=1e-8):
        super().__init__()
        self.bert_output = BertModel.from_pretrained(bert_pretrained_name)
        self.pool = nn.AvgPool1d(kernel_size=pool_stride, stride=pool_stride)
        self.cosin = nn.CosineSimilarity(dim=-1, eps=cosin_eps)

    def forward(self, input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2):
        output1 = self.bert_output(input_ids=input_ids1, token_type_ids=token_type_ids1, attention_mask=attention_mask1)
        pooler_output1 = output1.pooler_output
        #  pooled_output1 = self.pool(output1)
        output2 = self.bert_output(input_ids=input_ids2, token_type_ids=token_type_ids2, attention_mask=attention_mask2)
        pooler_output2 = output2.pooler_output
        # pooled_output2 = self.pool(output2)
        cosin_output = self.cosin(pooler_output1, pooler_output2)
        return cosin_output


class SentenceSimTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:

    # model_name_or_path: str = dataclasses.field(metadata={"help": "Path to model"})

    pool_stride: str = dataclasses.field(metadata={"help": "arg `pool_stride` of SBert"})

    cosin_eps: Union[int, float] = dataclasses.field(metadata={"help": "arg `cosin_eps` of SBert"})

    tokenizer_max_length: int = dataclasses.field(metadata={"help": "arg `max_length` of tokenizer"})


@dataclass
class DataTrainArguments:

    task_name: str = dataclasses.field(metadata={"help": "task name"})

    author: str = dataclasses.field(default="earlee")

    max_val_samples: int = dataclasses.field(default=9999999, metadata={"help": "I also don't know what's the meaning of the field"})


@dataclass
class DataCollator:

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None

    def __call__(self, features) -> dict:
        input_ids1 = []
        attention_mask1 = []
        token_type_ids1 = []
        input_ids2 = []
        attention_mask2 = []
        token_type_ids2 = []
        # added_special_token: dict = self.tokenizer.added_tokens_encoder

        for feat in features:
            logger.info(feat)
            text1, text2 = feat['text'].split(SpecialToken.SIMSEP.value)
            encode_dict1 = self.tokenizer(
                text1,
                add_special_tokens=True,
                truncation='longest_first',
                max_length=self.max_length,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )
            encode_dict2 = self.tokenizer(
                text2,
                add_special_tokens=True,
                truncation='longest_first',
                max_length=self.max_length,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )

            input_ids1.append(encode_dict1['input_ids'])
            attention_mask1.append(encode_dict1['attention_mask'])
            token_type_ids1.append(encode_dict1['token_type_ids'])

            input_ids2.append(encode_dict2['input_ids'])
            attention_mask2.append(encode_dict2['attention_mask'])
            token_type_ids2.append(encode_dict2['token_type_ids'])

        input_ids1 = torch.cat(input_ids1, dim=0)
        attention_mask1 = torch.cat(attention_mask1, dim=0)
        token_type_ids1 = torch.cat(token_type_ids1, dim=0)
        input_ids2 = torch.cat(input_ids2, dim=0)
        attention_mask2 = torch.cat(attention_mask2, dim=0)
        token_type_ids2 = torch.cat(token_type_ids2, dim=0)

        return {
            'input_ids1': input_ids1,
            'attention_mask1': attention_mask1,
            'token_type_ids1': token_type_ids1,
            'input_ids2': input_ids2,
            'attention_mask2': attention_mask2,
            'token_type_ids2': token_type_ids2
        }



def main(debug_json=None):
    parser = HfArgumentParser((DataTrainArguments, TrainingArguments, ModelArguments))
    if debug_json is not None:
        data_args, training_args, model_args = parser.parse_json_file(json_file=debug_json)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args, training_args, model_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters:  {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = BertTokenizerFast.from_pretrained('/Users/lichunyu/Desktop/EarleeNLP/bert-base-chinese')
    tokenizer.add_special_tokens({'additional_special_tokens':[SpecialToken.SIMSEP.value]})

    pool_stride, cosin_eps = model_args.pool_stride, model_args.cosin_eps
    model = SBert(bert_pretrained_name='/Users/lichunyu/Desktop/EarleeNLP/bert-base-chinese', pool_stride=pool_stride, cosin_eps=cosin_eps)


    def compute_metrics(p):
        """[tmp test]

        :param p: [description]
        :type p: [type]
        :return: [description]
        :rtype: [type]
        """        
        return {
                "precision": 1.,
                "recall": 1.,
                "f1": 1.,
                "accuracy": 1.,
            }

    data_collator = DataCollator(
        tokenizer,
        max_length=model_args.tokenizer_max_length
    )

    def gen_fake_data():
        _data = {'text': ['今天是星期四<SIMSEP>今天是周四','今天是星期四<SIMSEP>今天是周四','今天是星期四<SIMSEP>今天是周四','今天是星期四<SIMSEP>今天是周四',\
                '今天是星期四<SIMSEP>今天是周四','今天是星期四<SIMSEP>今天是周四','今天是星期四<SIMSEP>今天是周四','今天是星期四<SIMSEP>今天是周四',\
                '今天是星期四<SIMSEP>今天是周四','今天是星期四<SIMSEP>今天是周四','今天是星期四<SIMSEP>今天是周四','今天是星期四<SIMSEP>今天是周四',\
                '今天是星期四<SIMSEP>今天是周四','今天是星期四<SIMSEP>今天是周四','今天是星期四<SIMSEP>今天是周四','今天是星期四<SIMSEP>今天是周四']}
        _df = pd.DataFrame(_data)
        _dataset = Dataset.from_pandas(_df)
        return _dataset

    dt = load_dataset('text', data_files={
        'train': '/Users/lichunyu/Desktop/EarleeNLP/data/sim_data.txt',
        'eval': '/Users/lichunyu/Desktop/EarleeNLP/data/sim_data.txt'
    })
    train_dataset = dt['train']
    test_dataset = dt['eval']
    eval_dataset = dt['eval']

    trainer = SentenceSimTrainer(
        model=model,
        args=training_args,
        # train_dataset=train_dataset if training_args.do_train else None,
        # eval_dataset=eval_dataset if training_args.do_eval else None,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        logger.info("**** do_train start ****")
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    # main()
    main('/Users/lichunyu/Desktop/EarleeNLP/args/sentence_sim.json')