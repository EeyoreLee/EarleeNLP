# -*- encoding: utf-8 -*-
'''
@create_time: 2021/07/15 09:35:53
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
from torch.nn.modules import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import pandas as pd
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
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
from models.bert_query_ner import BertQueryNER, BertQueryNerConfig
from plugin.FGM import FGM
from utils.mrc_ner_dataloader import get_dataloader
from loss.dice_loss import DiceLoss
from metrics.functional.query_span_f1 import query_span_f1
from loss.mrc_ner_dice_loss import DiceLoss as MRCDiceLoss
from utils.args import CustomizeArguments


logger = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


def tokenize_batch(
    df,
    tokenizer,
    max_length=510,
    query='query',
    context='context',
    start_position='start_position',
    end_position='end_position',
    span_position='spac_position',
    label='label',
    **kwargs,
):
    input_ids = []
    attention_masks = []
    token_type_ids = []
    start_position_list = []
    end_position_list = []


    for idx, row in df.iterrows():
        encoded_dict = tokenizer(
                            row[query]+'[SEP]'+row[context], 
                            add_special_tokens = True,
                            truncation='longest_first',
                            max_length = max_length,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_token_type_ids=True,
                            return_tensors = 'pt',
                       )


        input_ids.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict['token_type_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df[label].tolist())

    return input_ids, attention_masks, labels


def gen_dataloader(df=None, df_train=None, df_eval=None, tokenizer=None, per_device_train_batch_size=None, per_device_eval_batch_size=None, test_size=0.2, **kwargs):
    if df is not None:
        df_train, df_eval = train_test_split(df, test_size=test_size)
    train_input_ids, train_attention_masks, train_labels = tokenize_batch(df_train, tokenizer, **kwargs)
    eval_input_ids, eval_attention_masks, eval_labels = tokenize_batch(df_eval, tokenizer,**kwargs)
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    eval_dataset = TensorDataset(eval_input_ids, eval_attention_masks, eval_labels)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=per_device_eval_batch_size)
    return train_dataloader, eval_dataloader


def compute_loss(_loss, start_logits, end_logits, span_logits,
                    start_labels, end_labels, match_labels, start_label_mask, end_label_mask, loss_type='dice', span_loss_candidates='all_'):
    batch_size, seq_len = start_logits.size()

    start_float_label_mask = start_label_mask.view(-1).float()
    end_float_label_mask = end_label_mask.view(-1).float()
    match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
    match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
    match_label_mask = match_label_row_mask & match_label_col_mask
    match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

    if span_loss_candidates == "all":
        # naive mask
        float_match_label_mask = match_label_mask.view(batch_size, -1).float()
    else:
        # use only pred or golden start/end to compute match loss
        start_preds = start_logits > 0
        end_preds = end_logits > 0
        if span_loss_candidates == "gold":
            match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
        else:
            match_candidates = torch.logical_or(
                (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                    & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                    & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
            )
        match_label_mask = match_label_mask & match_candidates
        float_match_label_mask = match_label_mask.view(batch_size, -1).float()
    if loss_type == "bce":
        bce_loss = _loss
        start_loss = bce_loss(start_logits.view(-1), start_labels.view(-1).float())
        start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
        end_loss = bce_loss(end_logits.view(-1), end_labels.view(-1).float())
        end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
        match_loss = bce_loss(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
        match_loss = match_loss * float_match_label_mask
        match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)
    else:
        dice_loss = _loss
        start_loss = dice_loss(start_logits, start_labels.float(), start_float_label_mask)
        end_loss = dice_loss(end_logits, end_labels.float(), end_float_label_mask)
        match_loss = dice_loss(span_logits, match_labels.float(), float_match_label_mask)

    return start_loss, end_loss, match_loss


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

    logger.info('Description: {}'.format(custom_args.description))
    if json_path:
        logger.info('json file path is : {}'.format(json_path))
        logger.info('json file args are: \n'+open(json_path, 'r').read())

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

    config = BertQueryNerConfig.from_pretrained(
        custom_args.config_name_or_path if custom_args.config_name_or_path else custom_args.model_name_or_path,
        # num_labels=custom_args.num_labels
    )

    model = BertQueryNER.from_pretrained(
        custom_args.model_name_or_path,
        config=config
    )

    tokenizer = BertTokenizer.from_pretrained(
        custom_args.tokenizer_name_or_path if custom_args.tokenizer_name_or_path else custom_args.model_name_or_path,
    )

    # data = pd.read_pickle(custom_args.pickle_data_path)
    # # df_train = pd.read_pickle(custom_args.train_pickle_data_path)
    # # df_eval = pd.read_pickle(custom_args.eval_pickle_data_path)
    # train_dataloader, eval_dataloader = gen_dataloader(
    #     df=data,
    #     # df_train=df_train,
    #     # df_eval=df_eval,
    #     tokenizer=tokenizer,
    #     per_device_train_batch_size=training_args.per_device_train_batch_size,
    #     per_device_eval_batch_size=training_args.per_device_eval_batch_size,
    #     test_size=custom_args.test_size,
    #     max_length=custom_args.max_length,
    # )

    train_dataloader = get_dataloader('train', 64)
    eval_dataloader = get_dataloader('test', 32)
    extra_loss = BCEWithLogitsLoss(reduction="none")
    extra_dice_loss = MRCDiceLoss(with_logits=True)

    # device = training_args.device if torch.cuda.is_available() else 'cpu'

    model = nn.DataParallel(model)
    model = model.cuda()
    total_bt = time.time()

    optimizer = AdamW(model.parameters(),
                  lr = 1e-5,
                  eps = 1e-8
                )

    total_steps = len(train_dataloader) * training_args.num_train_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 5, 
                                                num_training_steps = total_steps)

    weight_sum = custom_args.weight_start + custom_args.weight_end + custom_args.weight_span
    weight_start = custom_args.weight_start / weight_sum
    weight_end = custom_args.weight_end / weight_sum
    weight_span = custom_args.weight_span / weight_sum
    # fgm = FGM(model)

    for e in range(training_args.num_train_epochs):

        logger.info('============= Epoch {:} / {:} =============='.format(e + 1, training_args.num_train_epochs))
        logger.info('Training...')

        bt = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            # break

            if step % 50 == 0 and not step == 0:
                elapsed = format_time(time.time() - bt)
                logger.info('  Batch {:>5,}  of  {:>5,}.   Elapsed: {:}.    loss: {}'.format(step, len(train_dataloader), elapsed, total_train_loss/step))

            input_ids = batch[0].cuda()
            token_type_ids = batch[1].cuda()
            start_labels = batch[2].cuda()
            end_labels = batch[3].cuda()
            start_label_mask = batch[4].cuda()
            end_label_mask = batch[5].cuda()
            match_labels = batch[6].cuda()
            # sample_idx = batch[7].cuda()
            label_idx = batch[7].cuda()

            attention_mask = (input_ids != 0).long()

            model.zero_grad()

            start_logits, end_logits, span_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            start_loss, end_loss, match_loss = compute_loss(
                                                            _loss=extra_dice_loss,
                                                            start_logits=start_logits,
                                                            end_logits=end_logits,
                                                            span_logits=span_logits,
                                                            start_labels=start_labels,
                                                            end_labels=end_labels,
                                                            match_labels=match_labels,
                                                            start_label_mask=start_label_mask,
                                                            end_label_mask=end_label_mask
                                                            )

            loss = weight_start * start_loss + weight_end * end_loss + weight_span * match_loss

            # loss = output.loss
            # logits = output.logits
            total_train_loss += loss.item()

            loss.backward()

            # fgm.attack(epsilon=1.2)
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
            # if step % 50 == 0 and step != 0:
            #     break

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
            token_type_ids = batch[1].cuda()
            start_labels = batch[2].cuda()
            end_labels = batch[3].cuda()
            start_label_mask = batch[4].cuda()
            end_label_mask = batch[5].cuda()
            match_labels = batch[6].cuda()
            # sample_idx = batch[7].cuda()
            label_idx = batch[7].cuda()

            attention_mask = (input_ids != 0).long()

            with torch.no_grad():
                start_logits, end_logits, span_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

            start_loss, end_loss, match_loss = compute_loss(
                                                _loss=extra_dice_loss,
                                                start_logits=start_logits,
                                                end_logits=end_logits,
                                                span_logits=span_logits,
                                                start_labels=start_labels,
                                                end_labels=end_labels,
                                                match_labels=match_labels,
                                                start_label_mask=start_label_mask,
                                                end_label_mask=end_label_mask
                                                )

            loss = weight_start * start_loss + weight_end * end_loss + weight_span * match_loss

            total_eval_loss += loss.item()
            start_preds, end_preds = start_logits > 0, end_logits > 0
            eval_f1 = query_span_f1(start_preds, end_preds, span_logits, start_label_mask, end_label_mask, match_labels)
            # logger.info('eval_f1 : {}'.format(eval_f1))
            total_eval_f1 += eval_f1
            # break


        # logger.info(f'\n{classification_report(total_eval_p, total_eval_l, zero_division=1)}')
        avg_val_f1 = total_eval_f1 / len(eval_dataloader)
        # avg_val_acc = total_eval_acc / len(eval_dataloader)
        logger.info('F1: {0:.2f}'.format(avg_val_f1))
        # logger.info('Acc: {0:.2f}'.format(avg_val_acc))

        avg_val_loss = total_eval_loss / len(eval_dataloader)
        validation_time = format_time(time.time() - bt)

        logger.info('Validation Loss: {0:.2f}'.format(avg_val_loss))
        logger.info('Validation took: {:}'.format(validation_time))

        current_ckpt = training_args.output_dir + '/bert-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '-f1_' + str(int(avg_val_f1*100)) + '.pth'
        logger.info('Start to save checkpoint named {}'.format(current_ckpt))
        if custom_args.deploy is True:
            logger.info('>>>>>>>>>>>> saving the model <<<<<<<<<<<<<<')
            torch.save(model.module, current_ckpt)
        else:
            logger.info('>>>>>>>>>>>> saving the state_dict of model <<<<<<<<<<<<<')
            torch.save(model.module.state_dict(), current_ckpt)



if __name__ == '__main__':
    main('/root/EarleeNLP/args/mrc_ner.json')