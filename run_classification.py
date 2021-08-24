# -*- encoding: utf-8 -*-
'''
@create_time: 2021/07/07 13:59:47
@author: lichunyu
'''


from dataclasses import dataclass, field
import logging
import datetime
import time
import uuid
import os
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import pandas as pd
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    BertForSequenceClassification,
    BertTokenizer,
    BertConfig,
    AdamW,
    get_linear_schedule_with_warmup
)
from transformers.trainer_utils import is_main_process, get_last_checkpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

from models.BertForClassificationByDice import BertForClassificationByDice
from plugin.FGM import FGM


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

logger = logging.getLogger(__name__)


@dataclass
class CustomizeArguments:

    author: str = field(default="earlee", metadata={"help": "author"})

    model_name_or_path: str = field(default='', metadata={"help": "path of PTM"})

    config_name_or_path: str = field(default='')

    tokenizer_name_or_path: str = field(default='')

    num_labels: int = field(default=-1)

    pickle_data_path: str = field(default='')

    test_size: float = field(default=0.2)

    max_length: int = field(default=510)

    deploy: bool = field(default=True, metadata={"help": "save the state_dict of the model if set the field is false"})

    train_pickle_data_path: str = field(default='')

    eval_pickle_data_path: str = field(default='')

    log_file_path: str = field(default='logs/log.log')


def tokenize_batch(df, tokenizer, max_length=510, text_name='text', label_name='label', **kwargs):
    input_ids = []
    attention_masks = []
    for idx, row in df.iterrows():
        encoded_dict = tokenizer(
                            row[text_name],
                            # row['pair'],
                            add_special_tokens = True,
                            truncation='longest_first',
                            max_length = max_length,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt',
                       )


        input_ids.append(encoded_dict['input_ids'])

        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df[label_name].tolist())

    return input_ids, attention_masks, labels


def gen_dataloader(df=None, df_train=None, df_eval=None, tokenizer=None, per_device_train_batch_size=None, \
                per_device_eval_batch_size=None, test_size=0.2, label_name='label', **kwargs):
    if df is not None:
        df_train, df_eval, _, _ = train_test_split(df, df[label_name], test_size=test_size, stratify=df[label_name])
    train_input_ids, train_attention_masks, train_labels = tokenize_batch(df_train, tokenizer, **kwargs)
    eval_input_ids, eval_attention_masks, eval_labels = tokenize_batch(df_eval, tokenizer,**kwargs)
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    eval_dataset = TensorDataset(eval_input_ids, eval_attention_masks, eval_labels)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=per_device_eval_batch_size)
    return train_dataloader, eval_dataloader


def flat_f1(preds, labels):

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, pred_flat, average='micro')


def flat_acc(p, l):
    p_f = np.argmax(p, axis=1).flatten()
    l_f = l.flatten()
    return accuracy_score(l_f, p_f)


def format_time(elapsed):

    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def main(json_path=''):

    parser = HfArgumentParser((CustomizeArguments, TrainingArguments))

    if json_path:
        custom_args, training_args = parser.parse_json_file(json_file=json_path)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        custom_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        custom_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(custom_args.log_file_path)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )


    # last_checkpoint = None
    # if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #     if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
    #         raise ValueError(
    #             f"Output directory ({training_args.output_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #     elif last_checkpoint is not None:
    #         logger.info(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )

    set_seed(training_args.seed)

    config = BertConfig.from_pretrained(
        custom_args.config_name_or_path if custom_args.config_name_or_path else custom_args.model_name_or_path,
        num_labels=custom_args.num_labels
    )

    model = BertForSequenceClassification.from_pretrained(
        custom_args.model_name_or_path,
        config=config
    )

    tokenizer = BertTokenizer.from_pretrained(
        custom_args.tokenizer_name_or_path if custom_args.tokenizer_name_or_path else custom_args.model_name_or_path,
    )

    # config = RobertaConfig.from_pretrained(
    #     custom_args.config_name_or_path if custom_args.config_name_or_path else custom_args.model_name_or_path,
    #     num_labels=custom_args.num_labels
    # )

    # model = RobertaForSequenceClassification.from_pretrained(
    #     custom_args.model_name_or_path,
    #     config=config
    # )

    # tokenizer = RobertaTokenizerFast.from_pretrained(
    #     custom_args.tokenizer_name_or_path if custom_args.tokenizer_name_or_path else custom_args.model_name_or_path,
    # )

    data = pd.read_pickle(custom_args.pickle_data_path)
    # df_train = pd.read_pickle(custom_args.train_pickle_data_path)
    # df_eval = pd.read_pickle(custom_args.eval_pickle_data_path)
    train_dataloader, eval_dataloader = gen_dataloader(
        df=data,
        # df_train=df_train,
        # df_eval=df_eval,
        label_name='label',
        tokenizer=tokenizer,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        test_size=custom_args.test_size,
        max_length=custom_args.max_length,
    )

    device = training_args.device if torch.cuda.is_available() else 'cpu'

    model = nn.DataParallel(model)
    model = model.cuda()
    total_bt = time.time()

    optimizer = AdamW(model.parameters(),
                  lr = 5e-5,
                  eps = 1e-6
                )

    total_steps = len(train_dataloader) * training_args.num_train_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 2, 
                                                num_training_steps = total_steps)

    # fgm = FGM(model)

    for e in range(training_args.num_train_epochs):

        logger.info('============= Epoch {:} / {:} =============='.format(e + 1, training_args.num_train_epochs))
        logger.info('Training...')

        bt = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):

            if step % 50 == 0 and not step == 0:
                elapsed = format_time(time.time() - bt)
                logger.info('  Batch {:>5,}  of  {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            input_ids = batch[0].cuda()
            attention_mask = batch[1].cuda()
            labels = batch[2].cuda()

            model.zero_grad()

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = output.loss
            loss = loss.mean()
            logits = output.logits
            total_train_loss += loss.item()

            loss.backward()

            # fgm.attack(epsilon=0.9)
            # output_adv = model(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     labels=labels
            # )

            # loss_adv = output_adv.loss
            # loss_adv.backward()
            # fgm.restore()

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - bt)
        logger.info('Average training loss: {0:.2f}'.format(avg_train_loss))
        logger.info('Training epcoh took: {:}'.format(training_time))

        logger.info('Running Validation...')
        bt = time.time()
        model.eval()

        total_eval_loss = 0
        total_eval_f1 = 0
        total_eval_acc = 0
        total_eval_p = []
        total_eval_l = []

        for batch in eval_dataloader:

            input_ids = batch[0].cuda()
            attention_mask = batch[1].cuda()
            labels = batch[2].cuda()

            with torch.no_grad():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = output.loss
                loss = loss.mean()
                logits = output.logits

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            total_eval_f1 += flat_f1(logits, label_ids)
            total_eval_acc += flat_acc(logits, label_ids)
            total_eval_p.extend(np.argmax(logits, axis=-1).flatten().tolist())
            total_eval_l.extend(label_ids.flatten().tolist())

        logger.info(f'\n{classification_report(total_eval_p, total_eval_l, zero_division=1)}')
        avg_val_f1 = total_eval_f1 / len(eval_dataloader)
        avg_val_acc = total_eval_acc / len(eval_dataloader)
        logger.info('F1: {0:.2f}'.format(avg_val_f1))
        logger.info('Acc: {0:.2f}'.format(avg_val_acc))

        avg_val_loss = total_eval_loss / len(eval_dataloader)
        validation_time = format_time(time.time() - bt)

        logger.info('Validation Loss: {0:.2f}'.format(avg_val_loss))
        logger.info('Validation took: {:}'.format(validation_time))

        current_ckpt = training_args.output_dir + '/bert-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '-f1_' + str(int(avg_val_f1*100)) + '.pth'
        logger.info('Start to save checkpoint named {}'.format(current_ckpt))
        if custom_args.deploy is True:
            logger.info('>>>>>>>>>>>> saving the model <<<<<<<<<<<<<<')
            # torch.save(model.module, current_ckpt)
            if isinstance(model, nn.DataParallel):
                torch.save(model.module, current_ckpt)
            else:
                torch.save(model, current_ckpt)
        else:
            logger.info('>>>>>>>>>>>> saving the state_dict of model <<<<<<<<<<<<<')
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), current_ckpt)
            else:
                torch.save(model.state_dict(), current_ckpt)



if __name__ == '__main__':
    # main('/root/EarleeNLP/args/cls.json')
    # main('/root/EarleeNLP/args/df_intent_ood_roberta.json')
    main('/root/EarleeNLP/args/df_intent_hfl_roberta.json')
    # main('/root/EarleeNLP/args/news.json')
    # main('/root/EarleeNLP/args/df_intent_ood.json')
    # main('/root/EarleeNLP/args/df_intent_ood_roberta_6000.json')