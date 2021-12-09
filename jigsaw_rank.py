# -*- encoding: utf-8 -*-
'''
@create_time: 2021/12/09 15:13:38
@author: lichunyu
'''

from logging import getLogger
import logging
import sys
import os

import pandas as pd
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch
from torch.optim import SGD, AdamW
from tqdm import tqdm

from utils.dataloader.bertrank_dataloader import RankDataset
from utils.dataloader.regress_dataloader import JigsawValDataset
from models.BertRank import BertRank
from utils.common import date_now

COMMENT_DIR = '/ai/223/person/lichunyu/datasets/kaggle/jigsaw/comment'
COMMENT_MODEL_DIR = '/ai/223/person/lichunyu/models/kaggle/jigsaw-fourth/comment'
PreTrainModelPath = '/ai/223/person/lichunyu/pretrain-models/unitary_toxic-bert'

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
logger = getLogger(__name__)

def main():
    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('logs/log'+date_now()+'.log')],
    )
    logger.setLevel(logging.INFO)

    epoch = 10
    batch_size = 64
    data = pd.read_pickle(os.path.join(COMMENT_DIR, 'rank_train.pkl'))
    # val_data = pd.read_csv('/ai/223/person/lichunyu/datasets/kaggle/jigsaw/rate/validation_data.csv')
    val_data = pd.read_pickle(os.path.join(COMMENT_DIR, 'rank_val.pkl'))
    tokenizer = BertTokenizer.from_pretrained(PreTrainModelPath)

    model = BertRank(model_name_or_path=PreTrainModelPath)
    dataset = RankDataset(data, tokenizer, higher_text_colname='more_toxic', lower_text_colname='less_toxic')
    less_val_dataset = JigsawValDataset(val_data, 'less_toxic', tokenizer)
    more_val_dataset = JigsawValDataset(val_data, 'more_toxic', tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    less_val_dataloader = DataLoader(less_val_dataset, batch_size=batch_size)
    more_val_dataloader = DataLoader(more_val_dataset, batch_size=batch_size)
    # optimizer = SGD(model.parameters(), lr=4e-4, weight_decay=2)
    optimizer = AdamW(
        [
            {'params': model.bert.parameters(), 'lr': 5e-5},
        ],
        lr=5e-5,
    )

    model.cuda()

    for e in range(epoch):

        model.train()
        train_total_loss = 0
        step = 0
        for n, batch in enumerate(tqdm(train_dataloader)):
            model.zero_grad()
            step += 1
            input_ids = batch[0].cuda()
            attention_mask = batch[1].cuda()
            lower_input_ids = batch[2].cuda()
            lower_attention_mask = batch[3].cuda()
            target = batch[4].cuda()
            model_output = model(input_ids, attention_mask, lower_input_ids, lower_attention_mask, target)
            loss = model_output['loss']
            train_total_loss += loss.item()
            if (n % 50) == 0:
                logger.info(f'the loss of batch {n} is {loss.item()}')
            loss.backward()
            optimizer.step()

        logger.info('train step loss is {}'.format(train_total_loss/step))


        model.eval()
        less_toxic_scores = np.array([])
        more_toxic_scores = np.array([])
        for batch in tqdm(less_val_dataloader):
            input_ids = batch[0].cuda()
            attention_mask = batch[1].cuda()
            with torch.no_grad():
                model_output = model(input_ids, attention_mask)
                score = model_output['metric']
                score = score.detach().clone().cpu().numpy().flatten()
                less_toxic_scores = np.append(less_toxic_scores, score)

        for batch in tqdm(more_val_dataloader):
            input_ids = batch[0].cuda()
            attention_mask = batch[1].cuda()
            with torch.no_grad():
                model_output = model(input_ids, attention_mask)
                score = model_output['metric']
                score = score.detach().clone().cpu().numpy().flatten()
                more_toxic_scores = np.append(more_toxic_scores, score)

        acc_item = (less_toxic_scores < more_toxic_scores).sum()
        logger.info(f'~~~~~~ Acc item is {acc_item}  ~~~~~~~')
        acc = acc_item / len(less_toxic_scores)
        logger.info(f'~~~~~~ Acc score is {acc}  ~~~~~~~')

        current_ckpt = os.path.join(COMMENT_MODEL_DIR, f'bertrank-epoch-{e}-acc-{acc}.pth')
        torch.save(model.state_dict(), current_ckpt)


if __name__ == '__main__':
    main()
