# -*- encoding: utf-8 -*-
'''
@create_time: 2021/08/24 14:22:27
@author: lichunyu
'''



from dataclasses import dataclass, field
import logging
import datetime
import time
import uuid
import os
import sys
import datasets

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
    BertForTokenClassification,
    BertTokenizer,
    BertConfig,
    AdamW,
    get_linear_schedule_with_warmup
)
from transformers.trainer_utils import is_main_process, get_last_checkpoint
from sklearn.model_selection import train_test_split
from datasets import load_metric

from models.BertForClassificationByDice import BertForClassificationByDice
from plugin.FGM import FGM
from utils.args import CustomizeArguments
from utils.common import format_time, seq_idx2label


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

logger = logging.getLogger(__name__)



LABEL2IDX = {'O': 0,
        'I-Video-Play-name': 1,
        'I-Radio-Listen-channel': 2,
        'I-Calendar-Query-datetime_date': 3,
        'I-Alarm-Update-notes': 4,
        'I-FilmTele-Play-name': 5,
        'I-Alarm-Update-datetime_time': 6,
        'I-Radio-Listen-name': 7,
        'I-Alarm-Update-datetime_date': 8,
        'I-HomeAppliance-Control-appliance': 9,
        'I-Travel-Query-destination': 10,
        'I-HomeAppliance-Control-details': 11,
        'I-Radio-Listen-frequency': 12,
        'I-Music-Play-song': 13,
        'I-Weather-Query-datetime_date': 14,
        'B-Calendar-Query-datetime_date': 15,
        'I-Weather-Query-city': 16,
        'B-HomeAppliance-Control-appliance': 17,
        'B-Travel-Query-destination': 18,
        'I-Video-Play-datetime_date': 19,
        'I-Music-Play-artist': 20,
        'B-Alarm-Update-datetime_date': 21,
        'B-Weather-Query-city': 22,
        'B-Video-Play-name': 23,
        'B-Weather-Query-datetime_date': 24,
        'B-Alarm-Update-datetime_time': 25,
        'I-FilmTele-Play-artist': 26,
        'B-Alarm-Update-notes': 27,
        'B-HomeAppliance-Control-details': 28,
        'B-FilmTele-Play-name': 29,
        'B-Radio-Listen-channel': 30,
        'I-Music-Play-age': 31,
        'I-FilmTele-Play-age': 32,
        'B-Radio-Listen-name': 33,
        'I-FilmTele-Play-tag': 34,
        'I-Music-Play-album': 35,
        'B-Music-Play-artist': 36,
        'B-FilmTele-Play-artist': 37,
        'I-Travel-Query-departure': 38,
        'B-Music-Play-song': 39,
        'I-FilmTele-Play-play_setting': 40,
        'I-Travel-Query-datetime_date': 41,
        'B-Travel-Query-departure': 42,
        'I-Radio-Listen-artist': 43,
        'B-FilmTele-Play-tag': 44,
        'I-Travel-Query-datetime_time': 45,
        'B-Radio-Listen-frequency': 46,
        'B-Radio-Listen-artist': 47,
        'I-Video-Play-datetime_time': 48,
        'B-Video-Play-datetime_date': 49,
        'B-Travel-Query-datetime_date': 50,
        'I-FilmTele-Play-region': 51,
        'B-FilmTele-Play-region': 52,
        'B-FilmTele-Play-play_setting': 53,
        'I-TVProgram-Play-name': 54,
        'B-FilmTele-Play-age': 55,
        'B-Travel-Query-datetime_time': 56,
        'B-Music-Play-age': 57,
        'B-Music-Play-album': 58,
        'I-Video-Play-region': 59,
        'B-Video-Play-region': 60,
        'I-Music-Play-instrument': 61,
        'I-Weather-Query-datetime_time': 62,
        'I-TVProgram-Play-channel': 63,
        'B-Music-Play-instrument': 64,
        'I-Audio-Play-name': 65,
        'B-Video-Play-datetime_time': 66,
        'B-Weather-Query-datetime_time': 67,
        'B-TVProgram-Play-name': 68,
        'I-TVProgram-Play-datetime_date': 69,
        'I-Audio-Play-artist': 70,
        'B-Audio-Play-name': 71,
        'I-TVProgram-Play-datetime_time': 72,
        'B-TVProgram-Play-channel': 73,
        'B-TVProgram-Play-datetime_date': 74,
        'B-Audio-Play-artist': 75,
        'B-TVProgram-Play-datetime_time': 76,
        'I-Audio-Play-play_setting': 77,
        'B-Audio-Play-play_setting': 78,
        'B-Audio-Play-tag': 79,
        'I-Audio-Play-tag': 80}



def tokenize_batch(df, tokenizer, max_length=510, text_name='text', label_name='label', **kwargs):
    input_ids = []
    attention_masks = []
    labels = []
    for idx, row in df.iterrows():
        encoded_dict = tokenizer(
                            row[text_name],
                            add_special_tokens = True,
                            truncation='longest_first',
                            max_length = max_length,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt',
                       )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append([0]+row[label_name]+[0]+(max_length-2-len(row[label_name]))*[0])


    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.from_numpy(np.array(labels))

    return input_ids, attention_masks, labels


def gen_ner_dataloader(df=None, df_train=None, df_eval=None, tokenizer=None, per_device_train_batch_size=None, \
                per_device_eval_batch_size=None, test_size=0.2, label_name='label', **kwargs):
    if df is not None:
        df_train, df_eval, _, _ = train_test_split(df, df[label_name], test_size=test_size, stratify=df['seq_intent'])
    train_input_ids, train_attention_masks, train_labels = tokenize_batch(df_train, tokenizer, label_name=label_name, **kwargs)
    eval_input_ids, eval_attention_masks, eval_labels = tokenize_batch(df_eval, tokenizer, label_name=label_name, **kwargs)
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    eval_dataset = TensorDataset(eval_input_ids, eval_attention_masks, eval_labels)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=per_device_eval_batch_size)
    return train_dataloader, eval_dataloader



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

    set_seed(training_args.seed)

    config = BertConfig.from_pretrained(
        custom_args.config_name_or_path if custom_args.config_name_or_path else custom_args.model_name_or_path,
        num_labels=custom_args.num_labels
    )

    model = BertForTokenClassification.from_pretrained(
        custom_args.model_name_or_path,
        config=config
    )

    tokenizer = BertTokenizer.from_pretrained(
        custom_args.tokenizer_name_or_path if custom_args.tokenizer_name_or_path else custom_args.model_name_or_path,
    )

    data = pd.read_pickle(custom_args.pickle_data_path)

    train_dataloader, eval_dataloader = gen_ner_dataloader(
        df=data,
        label_name=custom_args.label_name,
        tokenizer=tokenizer,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        test_size=custom_args.test_size,
        max_length=custom_args.max_length,
    )

    metric_seqeval = load_metric('seqeval')

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

    seqeval_metric = datasets.load_metric('seqeval')

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
            mask = attention_mask.detach().cpu().numpy()

            pred = seq_idx2label(logits.argmax(axis=-1), mask, label2idx=LABEL2IDX)
            ref = seq_idx2label(label_ids, mask, label2idx=LABEL2IDX)
            seqeval_metric.add_batch(predictions=pred, references=ref)


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
    main('/root/EarleeNLP/args/cls.json')
