# -*- encoding: utf-8 -*-
'''
@create_time: 2021/07/23 15:22:53
@author: lichunyu
'''

import json
from collections import defaultdict
import os
import re
import time

import numpy as np
import torch.multiprocessing
from transformers import (
    BertTokenizer,
    RobertaTokenizerFast
)
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils.flat.base import load_ner
from utils.common import text_rm_space, bio_decode, viterbi_ensemble_decode
from models.flat_bert import load_yangjie_rich_pretrain_word_list, equip_chinese_ner_with_lexicon
from metircs.functional.f1_score import ner_extract
from run.run_ner import LABEL2IDX
from deploy.bert_modeling import bert_classification_inference


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


COMMAND2IDX = {'查询状态': 0,
                '开机': 1,
                '关机': 2,
                '音量调节': 3,
                '打开': 4,
                '模式调节': 5,
                '亮度调节': 6,
                '湿度调节': 7,
                '温度调节': 8,
                '风速调节': 9}


QUERY_TYPE2IDX = {'汽车票': 0,
                '导航': 1,
                '飞机票': 2,
                '船票查询': 3,
                '火车票': 4,
                '汽车票查询': 5,
                '机票查询': 6,
                '火车票查询': 7}


INTENT = ['Music-Play',
        'HomeAppliance-Control',
        'Travel-Query',
        'Calendar-Query',
        'FilmTele-Play',
        'Weather-Query',
        'Video-Play',
        'Alarm-Update',
        'TVProgram-Play',
        'Audio-Play',
        'Radio-Listen',
        'Other']


MRC_NER_LABEL = ['name',
                'datetime_date',
                'departure',
                'instrument',
                'datetime_time',
                'destination',
                'appliance',
                'notes',
                'details',
                'play_setting',
                'artist',
                'city',
                'frequency',
                'channel',
                'age',
                'album',
                'tag',
                'song']

#  第一版NER QUERY
QUERY_MAP = {
    'name': '用以识别某一个体或群体(人和事物)的专属名词',
    'datetime_date': '发生某一事情的确定的日子或时期',
    'departure': '完成一次出行所用的时间',
    'instrument': '科学技术上用于实验、计量、观测、检验、绘图等的器具或装置',
    'datetime_time': '物质的永恒运动、变化的持续性、顺序性的表现',
    'destination': '想要达到的地方',
    'appliance': '工作时所需用的器具',
    'notes': '听课、听报告、读书时所做的记录',
    'details': '起关键作用的小事',
    'play_setting': '播放设置',
    'artist': '杂技、戏曲、民间歌舞、曲艺演员',
    'city': '人口集中，居民以非农业人口为主，工商业比较发达的地区',
    'frequency': '每个对象出现的次数与总次数的比值',
    'channel': '电视台或电视网络',
    'age': '一个人从出生时起到计算时止生存的时间长度',
    'album': '期刊的某一期专门对某一个大家关注的领域、问题、事件进行集中报道、讨论',
    'tag': '具有相同特征的事物所形成的类别',
    'song': '由歌词和曲谱相结合的一种艺术形式',
    'region': '某一范围的地方'
}


I2L_ALARM_UPDATE = {0: '<pad>', 1: '<unk>', 2: 'O', 3: 'I-notes', 4: 'I-datetime_time', 5: 'I-datetime_date', 6: 'B-datetime_date', 7: 'B-datetime_time', 8: 'B-notes'}


SUB_CLS_LABEL = [
    'query_type',
    'play_mode',
    'command',
    'index',
    'language',
    'region',
    'type'
]


NER_CORRESPONDENCE = {
    'Travel-Query': ['datetime_date', 'departure', 'datetime_time', 'destination'],
    'Music-Play': ['instrument', 'artist', 'age', 'album', 'song'],
    # 'FilmTele-Play': ['name', 'artist', 'play_setting', 'age', 'region', 'tag'],
    'FilmTele-Play': ['name', 'artist', 'play_setting', 'age', 'tag'],
    # 'Video-Play': ['name', 'region', 'datetime_time', 'datetime_date'],
    'Video-Play': ['name', 'datetime_time', 'datetime_date'],
    'Radio-Listen': ['name', 'frequency', 'artist', 'channel'],
    'HomeAppliance-Control': ['appliance', 'details'],
    'Weather-Query': ['datetime_date', 'datetime_time', 'city'],
    'Alarm-Update': ['datetime_time', 'notes', 'datetime_date'],
    'Calendar-Query': ['datetime_date'],
    'TVProgram-Play': ['name', 'datetime_time', 'channel', 'datetime_date'],
    'Audio-Play': ['name', 'artist', 'play_setting', 'tag']
    }


CORRESPONDENCE = {
    'Travel-Query': ['datetime_date', 'departure', 'datetime_time', 'destination', 'query_type'],
    'Music-Play': ['language', 'instrument', 'play_mode', 'artist', 'age', 'album', 'song'],
    'FilmTele-Play': ['name', 'artist', 'play_setting', 'age', 'region', 'tag'],
    'Video-Play': ['name', 'region', 'datetime_time', 'datetime_date'],
    'Radio-Listen': ['name', 'frequency', 'artist', 'channel'],
    'HomeAppliance-Control': ['appliance', 'details', 'command'],
    'Weather-Query': ['index', 'datetime_date', 'datetime_time', 'city', 'type'],
    'Alarm-Update': ['datetime_time', 'notes', 'datetime_date'],
    'Calendar-Query': ['datetime_date'],
    'TVProgram-Play': ['name', 'datetime_time', 'channel', 'datetime_date'],
    'Audio-Play': ['name', 'language', 'artist', 'play_setting', 'tag']
    }

REGION_MAP = {
    '韩剧': '韩国',
    '美剧': '美国',
    '港片': '香港',
    '泰剧': '泰国',
    '内地': '大陆',
    '美片': '美国',
}


IDX2LABEL = {0: '<pad>', 1: '<unk>', 2: 'O', 3: 'I-name', 4: 'I-channel', 5: 'I-notes', 6: 'B-name', 7: 'I-artist', 8: 'I-appliance', 9: 'I-destination', 10: 'I-details', 11: 'I-frequency', 12: 'I-song', 13: 'B-artist', 14: 'I-age', 15: 'I-city', 16: 'B-appliance', 17: 'B-destination', 18: 'B-city', 19: 'B-notes', 20: 'B-details', 21: 'B-channel', 22: 'I-tag', 23: 'I-album', 24: 'I-play_setting', 25: 'I-departure', 26: 'B-song', 27: 'B-tag', 28: 'B-departure', 29: 'B-age', 30: 'B-frequency', 31: 'B-play_setting', 32: 'B-album', 33: 'I-instrument', 34: 'B-instrument'}


I2L_CALENDAR_QUERY = {0: '<pad>', 1: '<unk>', 2: 'O', 3: 'I-datetime_date', 4: 'B-datetime_date'}


I2L_FILMTELE_PLAY = {0: '<pad>', 1: '<unk>', 2: 'O', 3: 'I-FilmTele-Play-name', 4: 'I-FilmTele-Play-artist', 5: 'B-FilmTele-Play-name',
            6: 'I-FilmTele-Play-age', 7: 'I-FilmTele-Play-tag', 8: 'B-FilmTele-Play-artist', 9: 'B-FilmTele-Play-tag', 
            10: 'I-FilmTele-Play-play_setting', 11: 'B-FilmTele-Play-play_setting', 12: 'B-FilmTele-Play-age'}


I2L_DATE_AND_TIME = {0: '<pad>', 1: '<unk>', 2: 'O', 3: 'I-datetime_date', 4: 'I-datetime_time', 5: 'B-datetime_date', 6: 'B-datetime_time'}


class NERDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = self.dataset[index]
        row['chars'] = torch.tensor(row['chars'])
        row['target'] = torch.tensor(row['target'])
        row['bigrams'] = torch.tensor(row['bigrams'])
        row['seq_len'] = torch.tensor(row['seq_len'])
        row['lex_num'] = torch.tensor(row['lex_num'])
        row['lex_s'] = torch.tensor(row['lex_s'])
        row['lex_e'] = torch.tensor(row['lex_e'])
        row['lattice'] = torch.tensor(row['lattice'])
        row['pos_s'] = torch.tensor(row['pos_s'])
        row['pos_e'] = torch.tensor(row['pos_e'])
        return row


class NlpGoGo(object):

    def __init__(self, cls_model_path, cls_config_path, ner_model_path, ood_model_path, ood_bert_model_path=None, \
                ood_roberta_model_path=None, query_type_model_path=None, command_model_path=None, ood_bert_6000_model_path=None, \
                ood_roberta_6000_model_path=None, ood_fake_roberta_6000_model_path=None, ood_fake_roberta_2700_model_path=None, \
                ood_macbert_model_path=None, device='cuda', max_length=150, policy:dict=None, ood_config_path=None, cls_ext_model_path=None, \
                flat_alarm_update_model_path=None, flat_all_model_path=None, flat_device_map=None, ood_device_map=None, cls_device_map=None, \
                ood_hfl_model_path=None, flat_filmtele_play_model_path=None):
        """
        policy = {
                'cls_model': 'bert',
                'only_cls': False,
                    }
        """
        # ner flat detached model path
        self.flat_device_map = flat_device_map
        self.ood_device_map = ood_device_map
        self.cls_device_map = cls_device_map
        self.flat_alarm_update_model_path = flat_alarm_update_model_path
        self.flat_filmtele_play_model_path = flat_filmtele_play_model_path

        self.yangjie_rich_pretrain_word_path = '/root/pretrain-models/flat/ctb.50d.vec'
        self.yangjie_rich_pretrain_char_and_word_path = '/root/pretrain-models/flat/yangjie_word_char_mix.txt'
        self.device = device
        self.policy = policy
        # if policy['cls_model'] == 'bert':
        self.cls_model = torch.load(cls_model_path, map_location=torch.device(device))
        self.cls_ext_model = torch.load(cls_ext_model_path, map_location=torch.device(device))
        self.tokenizer = BertTokenizer.from_pretrained(cls_config_path)
        self.max_length = max_length

        self.hfl_ext_large_tokenzier = BertTokenizer.from_pretrained('/root/pretrain-models/hfl-chinese-roberta-wwm-ext-large')
        self.hfl_ext_base_tokenizer = BertTokenizer.from_pretrained('/root/pretrain-models/hfl_chinese-roberta-wwm-ext')
        self.macbert_large_tokenizer = BertTokenizer.from_pretrained('/root/pretrain-models/hfl-chinese-macbert-large')

        if policy['only_cls'] is False:
            if self.policy['ner'] == 'flat':
                # self.ner_model = torch.load('/ai/223/person/lichunyu/models/df/ner/flat-2021-08-26-06-15-37-f1_92.pth', map_location=torch.device('cuda'))
                # self.ner_model = torch.load(flat_all_model_path, map_location=self.flat_device_map['all'])
                self.ner_model = [
                    torch.load(flat_all_model_path, map_location=self.flat_device_map['all']),
                    torch.load('/ai/223/person/lichunyu/models/df/ner_detached/flat-2021-09-13-17-37-50-f1_90.pth', map_location=self.flat_device_map['all']),
                    torch.load('/ai/223/person/lichunyu/models/df/ner_detached/flat-2021-09-13-17-18-01-f1_89.pth', map_location=self.flat_device_map['all']),
                    torch.load('/ai/223/person/lichunyu/models/df/ner_detached/flat-2021-09-13-16-48-04-f1_90.pth', map_location=self.flat_device_map['all']),
                    # torch.load('/ai/223/person/lichunyu/models/df/ner_detached/flat-2021-09-13-16-08-47-f1_89.pth', map_location='cuda:2'),
                    # torch.load('/ai/223/person/lichunyu/models/df/ner_detached/flat-2021-09-13-16-48-04-f1_90.pth', map_location=self.flat_device_map['all']),
                ]
            else:
                m = torch.load(ner_model_path, map_location=torch.device(device))
                self.ner_model = m

        if policy['ood_model'] == 'roberta':
            self.ood_model = torch.load(ood_model_path, map_location=torch.device(device))
            self.ood_tokenizer = RobertaTokenizerFast.from_pretrained(ood_config_path)
        elif policy['ood_model'] == 'bert':
            self.ood_model = torch.load(ood_model_path, map_location=torch.device(device))
            self.ood_tokenizer = self.tokenizer
        else:
            self.ood_bert_model = torch.load(ood_bert_model_path, map_location=torch.device(device))
            self.ood_bert_tokenizer = self.tokenizer
            # self.ood_roberta_model = torch.load(ood_roberta_model_path, map_location=torch.device(device))
            # self.ood_roberta_tokenizer = RobertaTokenizerFast.from_pretrained(ood_config_path)
            self.ood_bert_6000_model = torch.load(ood_bert_6000_model_path, map_location=torch.device(device))
            # self.ood_roberta_6000_model = torch.load(ood_roberta_6000_model_path, map_location=torch.device(device))
            self.ood_fake_roberta_model = torch.load(ood_fake_roberta_6000_model_path, map_location=torch.device(device))
            self.ood_fake_roberta_2700_model = torch.load(ood_fake_roberta_2700_model_path, map_location=torch.device(device))
            self.ood_macbert_6000_model = torch.load(ood_macbert_model_path, map_location=torch.device(device))
            self.ood_hfl_model_6000_model = torch.load(ood_macbert_model_path, map_location='cuda:3')
            self.ood_bert_6000_qingyun_model = torch.load('/ai/223/person/lichunyu/models/df/best/bert-2021-09-08-16-08-12-f1_99.pth', map_location='cuda:3')
            self.ood_bert_qy_weibo_model = torch.load('/ai/223/person/lichunyu/models/df/best/bert-2021-09-08-17-40-16-f1_99.pth', map_location='cuda:4')
            self.ood_bert_all_tnew_model = torch.load('/ai/223/person/lichunyu/models/df/best/bert-2021-09-09-10-31-19-f1_99.pth', map_location='cuda:4')
            self.ood_bert_xhj_model = torch.load('/ai/223/person/lichunyu/models/df/best/bert-2021-09-09-10-30-46-f1_99.pth', map_location='cuda:4')

        self.query_type_model = torch.load(query_type_model_path, map_location=torch.device(device))
        self.command_model = torch.load(command_model_path, map_location=torch.device(device))

    def is_ood(self, text) -> bool:
        # return False
        id2label = {idx: label for idx, label in enumerate(INTENT)}
        if self.policy['ood_model'] == 'ensemble':
            self.classification_vote = defaultdict(float)

            bert_intent = bert_classification_inference(text, self.ood_bert_model, self.ood_bert_tokenizer, idx2label=id2label)
            bert_6000_intent = bert_classification_inference(text, self.ood_bert_6000_model, self.ood_bert_tokenizer, idx2label=id2label)
            hfl_bert_6000_large_intent = bert_classification_inference(text, self.ood_fake_roberta_model, self.hfl_ext_large_tokenzier, idx2label=id2label)
            hfl_bert_2700_large_intent = bert_classification_inference(text, self.ood_fake_roberta_2700_model, self.hfl_ext_large_tokenzier, idx2label=id2label)
            macbert_6000_intent = bert_classification_inference(text, self.ood_macbert_6000_model, self.macbert_large_tokenizer, idx2label=id2label)
            hfl_bert_6000_intent = bert_classification_inference(text, self.ood_hfl_model_6000_model, self.hfl_ext_base_tokenizer, idx2label=id2label, device='cuda:3')
            bert_6000_qingyun_intent = bert_classification_inference(text, self.ood_bert_6000_qingyun_model, self.hfl_ext_base_tokenizer, idx2label=id2label, device='cuda:3')
            bert_qy_weibo_intent = bert_classification_inference(text, self.ood_bert_qy_weibo_model, self.hfl_ext_base_tokenizer, idx2label=id2label, device='cuda:4')
            bert_all_tnew_intent = bert_classification_inference(text, self.ood_bert_all_tnew_model, self.hfl_ext_base_tokenizer, idx2label=id2label, device='cuda:4')
            bert_xhj_intent = bert_classification_inference(text, self.ood_bert_xhj_model, self.hfl_ext_base_tokenizer, idx2label=id2label, device='cuda:4')

            self.classification_vote[bert_intent] += 1
            self.classification_vote[bert_6000_intent] += 1
            self.classification_vote[hfl_bert_6000_large_intent] += 1
            self.classification_vote[hfl_bert_2700_large_intent] += 1
            self.classification_vote[macbert_6000_intent] += 1
            self.classification_vote[hfl_bert_6000_intent] += 1
            self.classification_vote[bert_6000_qingyun_intent] += 1
            self.classification_vote[bert_qy_weibo_intent] += 1
            self.classification_vote[bert_all_tnew_intent] += 1
            self.classification_vote[bert_xhj_intent] += 1
            # vote = sorted(vote.items(), key=lambda x: x[-1])
            # if vote[-1][0] != INTENT[-1]:
            #     return False
            # else:
            #     return True

            if INTENT[-1] in self.classification_vote:
                return True
            return False

        else:

            intent, logits = bert_classification_inference(text, self.ood_model, self.ood_bert_tokenizer, max_length=self.max_length, \
                idx2label=id2label, device='cuda')
            if torch.max(F.softmax(logits)).detach().cpu().numpy().tolist() < 0.65:
                intent = INTENT[-1]

            if intent != INTENT[-1]:
                return False
            return True

    def classify(self, text):
        # return INTENT[10]
        id2label = {idx: label for idx, label in enumerate(INTENT)}
        if self.policy['cls_model'] == 'bert':
            bert_base_intent = bert_classification_inference(text, self.cls_model, self.tokenizer, idx2label=id2label)

            return bert_base_intent

        elif self.policy['cls_model'] == 'ensemble':

            bert_base_intent = bert_classification_inference(text, self.cls_model, self.tokenizer, idx2label=id2label)
            self.classification_vote[bert_base_intent] += 1

            self.classification_vote = sorted(self.classification_vote.items(), key=lambda x: x[-1])
            return self.classification_vote[-1][0]


    def ner(self, text, classification):
        if self.policy['ner'] == 'mrc':
            input_ids = []
            attention_mask = []
            token_type_ids = []

            for slot in NER_CORRESPONDENCE[classification]:
                query = QUERY_MAP[slot]
                encoded_dict = self.tokenizer(
                                query,
                                normalization(text),
                                add_special_tokens = True,
                                truncation='longest_first',
                                max_length = self.max_length,
                                padding = 'max_length',
                                return_attention_mask = True,
                                return_token_type_ids=True,
                                return_tensors = 'pt',
                        )

                input_ids.append(encoded_dict['input_ids'])
                attention_mask.append(encoded_dict['attention_mask'])
                token_type_ids.append(encoded_dict['token_type_ids'])
            input_ids = torch.cat(input_ids, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)
            token_type_ids = torch.cat(token_type_ids, dim=0)

            self.ner_model.eval()
            with torch.no_grad():
                input_ids = input_ids.to(self.device).to(torch.int64)
                attention_mask = attention_mask.to(self.device).to(torch.int64)
                token_type_ids = token_type_ids.to(self.device).to(torch.int64)
                start_logits, end_logits, span_logits = self.ner_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

            res = {}
            for idx in range(len(NER_CORRESPONDENCE[classification])):
                slot = self.compute_ner(
                    text,
                    start_logits[idx],
                    end_logits[idx],
                    span_logits[idx],
                    token_type_ids[idx],
                    NER_CORRESPONDENCE[classification][idx]
                )
                if len(slot) == 0:
                    continue
                elif len(slot) == 1:
                    res[NER_CORRESPONDENCE[classification][idx]] = slot[0]
                else:
                    res[NER_CORRESPONDENCE[classification][idx]] = slot
            return res
        else:
            input_ids = []
            attention_mask = []
            token_type_ids = []
            encoded_dict = self.tokenizer(
                            normalization(text),
                            add_special_tokens = True,
                            truncation='longest_first',
                            max_length = self.max_length,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_token_type_ids=True,
                            return_tensors = 'pt',
                    )

            input_ids.append(encoded_dict['input_ids'])
            attention_mask.append(encoded_dict['attention_mask'])
            token_type_ids.append(encoded_dict['token_type_ids'])
            input_ids = torch.cat(input_ids, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)
            token_type_ids = torch.cat(token_type_ids, dim=0)
            self.ner_model.eval()
            with torch.no_grad():
                input_ids = input_ids.to(self.device).to(torch.int64)
                attention_mask = attention_mask.to(self.device).to(torch.int64)
                token_type_ids = token_type_ids.to(self.device).to(torch.int64)
                output = self.ner_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                logits = output.logits.detach().cpu().numpy()
                label_ids = logits.argmax(axis=-1)
                mask = attention_mask.detach().cpu().numpy()
                text, offset = text_rm_space(text)
            res = bio_decode(label_ids, text, mask, label2idx=LABEL2IDX)
            slots = flat_slot_clean(res)
            return slots

    def compute_ner(self, text, _start_logits, _end_logits, _span_logits, _token_type_ids, classification):
        if len(_start_logits.shape) == 1:
            _start_logits = _start_logits[None, :]
        if len(_end_logits.shape) == 1:
            _end_logits = _end_logits[None, :]
        if len(_span_logits.shape) == 2:
            _span_logits = _span_logits[None, :, :]
        if len(_token_type_ids.shape) == 1:
            _token_type_ids = _token_type_ids[None, :]

        res = []
        text_token_idx = torch.where(_token_type_ids == 1)[-1]
        start_text_token_idx = text_token_idx[0]
        end_text_token_idx = text_token_idx[-1]
        start_tag = torch.where(_start_logits[:,start_text_token_idx:end_text_token_idx+1] > 0)[-1]
        end_tag = torch.where(_end_logits[:,start_text_token_idx:end_text_token_idx+1] > 0)[-1]
        span = _span_logits[0][start_text_token_idx:end_text_token_idx+1][:,start_text_token_idx:end_text_token_idx+1]
        if len(end_tag) == 0:
            return res
        for t in start_tag:
            end = torch.argmax(span[t,end_tag])
            text_end_idx = end_tag[end]
            if text_end_idx < t:
                continue
            slot = text[t:text_end_idx+1]
            if classification == 'region':
                if slot in REGION_MAP:
                    slot = REGION_MAP[slot]
            res.append(text[t:text_end_idx+1])
        # print(res)
        return res

    def music_play_play_mode(self, text):
        cls_2 = [
            '顺序播放',
            '挨着',
            '挨个'
        ]
        for i in cls_2:
            if i in text:
                return '顺序播放'
        cls_0 = [
            '随机',
            '随便',
            '任意',
        ]
        for i in cls_0:
            if i in text:
                return '随机播放'
        cls_1 = [
            '单曲',
            '循环'
        ]
        for i in cls_1:
            if i in text:
                return '单曲循环'
        return ''

    def music_play_language(self, text):
        cls_0 = [
            '英文',
            '英语'
        ]
        for i in cls_0:
            if i in text:
                return '英语'
            
        cls_1 = [
            '韩语',
            '韩文'
        ]
        for i in cls_1:
            if i in text:
                return '韩语'
        if '粤语' in text:
            return '粤语'
        if '西班牙语' in text:
            return '西班牙语'
        if '藏语' in text:
            return '藏语'
        if '德语' in text:
            return '德语'
        if '日语' in text or '日文' in text:
            return '日语'
        if '越南语' in text:
            return '越南语'
        if '华语' in text or '中文' in text:
            return '华语'
        if '意大利语' in text:
            return '意大利语'
        if '法语' in text:
            return '法语'
        if '阿拉伯语' in text:
            return '阿拉伯语'
        if '俄语' in text or '俄文' in text:
            return '俄语'
        return ''

    def weather_query_index(self, text):
        if '紫外线' in text:
            return '紫外线指数'
        cls_5 = [
            '降水情况',
            '降雨情况',
            '降水量',
            '降雨量'
        ]
        for i in cls_5:
            if i in text:
                return '降水量'
        cls_0 = [
            '穿',
            '衣物'
        ]
        for i in cls_0:
            if i in text:
                return '穿衣指数'
        cls_3 = [
            '湿度'
        ]
        for i in cls_3:
            if i in text:
                return '湿度'
        cls_1 = [
            '气温',
            '温度',
            '冷',
            '度',
            '温',
            '热'
        ]
        for i in cls_1:
            if i in text:
                return '温度'
        cls_2 = [
            '空气',
            '雾霾'
        ]
        for i in cls_2:
            if i in text:
                return '空气质量'
        if '日出' in text:
            return '日出'
        cls_4 = [
            '风向',
            '北风',
            '东风',
            '西风',
            '南风',
        ]
        for i in cls_4:
            if i in text:
                return '风向'
        # if '风' in text:
        #     return '风力'
        return ''

    def region_extract(self, text):
        result = []  # (region, sub_text, index)
        cls_0 = [
            '美国',
            '美剧',
            '美片'
        ]
        for i in cls_0:
            if i in text:
                index = text.index(i)
                result.append(('美国', i, index))
        cls_1 = [
            '日本'
        ]
        for i in cls_1:
            if i in text:
                index = text.index(i)
                result.append(('日本', i, index))
        cls_2 = [
            '国产'
        ]
        for i in cls_2:
            if i in text:
                index = text.index(i)
                result.append(('国产', i, index))
        cls_3 = [
            '香港',
            '港片'
        ]
        for i in cls_3:
            if i in text:
                index = text.index(i)
                result.append(('香港', i, index))
        cls_4 = [
            '法国'
        ]
        for i in cls_4:
            if i in text:
                index = text.index(i)
                result.append(('法国', i, index))
        cls_5 = [
            '韩国',
            '韩剧'
        ]
        for i in cls_5:
            if i in text:
                index = text.index(i)
                result.append(('韩国', i, index))
        cls_6 = [
            '伦敦'
        ]
        for i in cls_6:
            if i in text:
                index = text.index(i)
                result.append(('伦敦', i, index))
        cls_7 = [
            '欧美'
        ]
        for i in cls_7:
            if i in text:
                index = text.index(i)
                result.append(('欧美', i, index))
        cls_8 = [
            '大陆',
            '内地'
        ]
        for i in cls_8:
            if i in text:
                index = text.index(i)
                result.append(('大陆', i, index))
        cls_9 = [
            '台湾'
        ]
        for i in cls_9:
            if i in text:
                index = text.index(i)
                result.append(('台湾', i, index))
        cls_10 = [
            '上海'
        ]
        for i in cls_10:
            if i in text:
                index = text.index(i)
                result.append(('上海', i, index))
        cls_11 = [
            '中国'
        ]
        for i in cls_11:
            if i in text:
                index = text.index(i)
                result.append(('中国', i, index))
        cls_12 = [
            '英国'
        ]
        for i in cls_12:
            if i in text:
                index = text.index(i)
                result.append(('英国', i, index))
        cls_13 = [
            '泰国',
            '泰剧'
        ]
        for i in cls_13:
            if i in text:
                index = text.index(i)
                result.append(('泰国', i, index))
        cls_14 = [
            '四川'
        ]
        for i in cls_14:
            if i in text:
                index = text.index(i)
                result.append(('四川', i, index))
        cls_15 = [
            '国创'
        ]
        for i in cls_15:
            if i in text:
                index = text.index(i)
                result.append(('国创', i, index))
        cls_16 = [
            '北京'
        ]
        for i in cls_16:
            if i in text:
                index = text.index(i)
                result.append(('北京', i, index))
        cls_17 = [
            '安徽'
        ]
        for i in cls_17:
            if i in text:
                index = text.index(i)
                result.append(('安徽', i, index))
        cls_18 = [
            '欧洲'
        ]
        for i in cls_18:
            if i in text:
                index = text.index(i)
                result.append(('欧洲', i, index))
        cls_19 = [
            '重庆'
        ]
        for i in cls_19:
            if i in text:
                index = text.index(i)
                result.append(('重庆', i, index))
        cls_20 = [
            '巴西'
        ]
        for i in cls_20:
            if i in text:
                index = text.index(i)
                result.append(('巴西', i, index))
        cls_21 = [
            '温哥华'
        ]
        for i in cls_21:
            if i in text:
                index = text.index(i)
                result.append(('温哥华', i, index))
        cls_22 = [
            '南京'
        ]
        for i in cls_22:
            if i in text:
                index = text.index(i)
                result.append(('南京', i, index))
        cls_23 = [
            '伊朗'
        ]
        for i in cls_23:
            if i in text:
                index = text.index(i)
                result.append(('伊朗', i, index))
        cls_24 = [
            '越南'
        ]
        for i in cls_24:
            if i in text:
                index = text.index(i)
                result.append(('越南', i, index))
        cls_25 = [
            '意大利'
        ]
        for i in cls_25:
            if i in text:
                index = text.index(i)
                result.append(('意大利', i, index))
        cls_26 = [
            '武汉'
        ]
        for i in cls_26:
            if i in text:
                index = text.index(i)
                result.append(('武汉', i, index))
        cls_27 = [
            '新加坡'
        ]
        for i in cls_27:
            if i in text:
                index = text.index(i)
                result.append(('新加坡', i, index))
        cls_28 = [
            '浙江'
        ]
        for i in cls_28:
            if i in text:
                index = text.index(i)
                result.append(('浙江', i, index))
        cls_29 = [
            '俄罗斯'
        ]
        for i in cls_29:
            if i in text:
                index = text.index(i)
                result.append(('俄罗斯', i, index))
        cls_30 = [
            '雅典'
        ]
        for i in cls_30:
            if i in text:
                index = text.index(i)
                result.append(('雅典', i, index))
        cls_31 = [
            '印度'
        ]
        for i in cls_31:
            if i in text:
                index = text.index(i)
                result.append(('印度', i, index))
        cls_32 = [
            '德国'
        ]
        for i in cls_32:
            if i in text:
                index = text.index(i)
                result.append(('德国', i, index))
        cls_33 = [
            '湖南'
        ]
        for i in cls_33:
            if i in text:
                index = text.index(i)
                result.append(('湖南', i, index))
        cls_34 = [
            '澳大利亚'
        ]
        for i in cls_34:
            if i in text:
                index = text.index(i)
                result.append(('澳大利亚', i, index))
        cls_35 = [
            '洛阳'
        ]
        for i in cls_35:
            if i in text:
                index = text.index(i)
                result.append(('洛阳', i, index))
        cls_36 = [
            '深圳'
        ]
        for i in cls_36:
            if i in text:
                index = text.index(i)
                result.append(('深圳', i, index))
        cls_37 = [
            '意大利'
        ]
        for i in cls_37:
            if i in text:
                index = text.index(i)
                result.append(('意大利', i, index))
        cls_38 = [
            '巴塞罗那'
        ]
        for i in cls_38:
            if i in text:
                index = text.index(i)
                result.append(('巴塞罗那', i, index))
        cls_39 = [
            '西安'
        ]
        for i in cls_39:
            if i in text:
                index = text.index(i)
                result.append(('西安', i, index))
        cls_40 = [
            '苏州'
        ]
        for i in cls_40:
            if i in text:
                index = text.index(i)
                result.append(('苏州', i, index))
        cls_41 = [
            '辽宁'
        ]
        for i in cls_41:
            if i in text:
                index = text.index(i)
                result.append(('辽宁', i, index))
        cls_42 = [
            '海口'
        ]
        for i in cls_42:
            if i in text:
                index = text.index(i)
                result.append(('海口', i, index))
        if result:
            result = sorted(result, key=lambda x: x[-1])
            return result[0][0]
        return ''


    def weather_query_type(self, text):
        cls_0 = [
            '雾',
            '能见度'
        ]
        for i in cls_0:
            if i in text:
                return '雾'
        if '多云' in text:
            return '多云'
        if '空气污染指数' in text:
            return '空气污染指数'
        # if '衣物' in text:
        #     return '穿衣指数'
        if '暴雨' in text:
            return '暴雨'
        cls_1 = [
            '雨',
            '降水'
        ]
        for i in cls_1:
            if i in text:
                return '雨'
        if '晴' in text:
            return '晴'
        if '暴雪' in text:
            return '暴雪'
        if '台风' in text:
            return '台风'
        if '风' in text:
            cls_4 = [
                '风向',
                '北风',
                '东风',
                '西风',
                '南风',
                ]
            for i in cls_4:
                if i in text:
                    return ''
            return '风'
        if '雪' in text:
            return '雪'
        if '泥石流' in text:
            return '泥石流'
        if '阴' in text:
            return '阴'
        if '冰雹' in text:
            return '冰雹'
        if '沙尘' in text:
            return '沙尘'
        if '霜' in text:
            return '霜'
        return ''

    def travel_query_query_type(self, text):
        input_ids = []
        attention_mask = []
        encoded_dict = self.tokenizer(
                        text, 
                        add_special_tokens = True,
                        truncation='longest_first',
                        max_length = self.max_length,
                        padding = 'max_length',
                        return_attention_mask = True,
                        return_tensors = 'pt',
                )

        input_ids.append(encoded_dict['input_ids'])
        attention_mask.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)

        self.query_type_model.eval()
        with torch.no_grad():
            input_ids = input_ids.to(self.device).to(torch.int64)
            attention_mask = attention_mask.to(self.device).to(torch.int64)
            output = self.query_type_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        id2query_type = {v: k for k, v in QUERY_TYPE2IDX.items()}
        return id2query_type[torch.argmax(output.logits).detach().cpu().numpy().tolist()]

    def home_appliance_control_command(self, text):
        input_ids = []
        attention_mask = []
        encoded_dict = self.tokenizer(
                        text, 
                        add_special_tokens = True,
                        truncation='longest_first',
                        max_length = self.max_length,
                        padding = 'max_length',
                        return_attention_mask = True,
                        return_tensors = 'pt',
                )

        input_ids.append(encoded_dict['input_ids'])
        attention_mask.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)

        self.command_model.eval()
        with torch.no_grad():
            input_ids = input_ids.to(self.device).to(torch.int64)
            attention_mask = attention_mask.to(self.device).to(torch.int64)
            output = self.command_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        id2query_type = {v: k for k, v in COMMAND2IDX.items()}
        return id2query_type[torch.argmax(output.logits).detach().cpu().numpy().tolist()]

    def flat_ner_all(self, real_text_list=None, intent='all', manager=None):
        if intent == 'all':
            _all_test_path = [
                '/ai/223/person/lichunyu/datasets/dataf/test/test_A_text.nonletter',
                '/ai/223/person/lichunyu/datasets/dataf/test/test_A_text.nonletter',
                '/ai/223/person/lichunyu/datasets/dataf/test/test_A_text.nonletter',
                '/ai/223/person/lichunyu/datasets/dataf/test/test_A_text.nonletter',
            ]
            _all_placeholder_path = [
                '/ai/223/person/lichunyu/datasets/dataf/test/train.nonletter',
                '/ai/223/person/lichunyu/datasets/dataf/test/train.nonletter',
                '/ai/223/person/lichunyu/datasets/dataf/test/train.nonletter',
                '/ai/223/person/lichunyu/datasets/dataf/test/train.nonletter',
            ]
            _train_path = [
                '/ai/223/person/lichunyu/datasets/dataf/seq_label/new_all_train_detached_clean.train',
                '/ai/223/person/lichunyu/datasets/dataf/seq_label/new_all_train_detached_clean.train',
                '/ai/223/person/lichunyu/datasets/dataf/seq_label/new_all_train_detached_clean.train',
                '/ai/223/person/lichunyu/datasets/dataf/seq_label/new_all_train_detached_clean.train',
            ]
            _dev_path = [
                '/ai/223/person/lichunyu/datasets/dataf/seq_label/new_all_train_detached_clean.test',
                '/ai/223/person/lichunyu/datasets/dataf/seq_label/new_all_train_detached_clean.test',
                '/ai/223/person/lichunyu/datasets/dataf/seq_label/new_all_train_detached_clean.test',
                '/ai/223/person/lichunyu/datasets/dataf/seq_label/new_all_train_detached_clean.test',
            ]
            # res = self._flat_ner_all(self.ner_model, real_text_list, with_placeholder=False, idx2label=IDX2LABEL, _device=self.flat_device_map[intent])
            res = self._flat_ner_all(self.ner_model, real_text_list, train_path=_train_path, dev_path=_dev_path, \
                _device=['cuda:1', 'cuda:1', 'cuda:1', 'cuda:1'], idx2label=IDX2LABEL, placeholder_path=_all_placeholder_path, \
                test_path=_all_test_path)
            if manager is not None:
                manager[intent] = res
            return res
            # return self._flat_ner_all(self.ner_model, real_text_list, with_placeholder=False, idx2label=IDX2LABEL)
        intent2model = {
            'Alarm-Update': [
                torch.load('/ai/223/person/lichunyu/models/df/ner/flat-2021-09-10-15-03-03-f1_97.pth', map_location=self.flat_device_map[intent]),
                ],
            # 'FilmTele-Play': torch.load(self.flat_filmtele_play_model_path, map_location=self.flat_device_map[intent]),
            'date_and_time': torch.load('/ai/223/person/lichunyu/models/df/best/flat-2021-09-08-11-24-01-f1_97.pth', map_location=self.flat_device_map[intent]),
                # torch.load('/ai/223/person/lichunyu/models/df/ner_detached/flat-2021-09-16-14-59-07-f1_98.pth', map_location=self.flat_device_map[intent]),
                # torch.load('/ai/223/person/lichunyu/models/df/ner_detached/flat-2021-09-16-14-54-40-f1_98.pth', map_location=self.flat_device_map[intent]),
                # torch.load('/ai/223/person/lichunyu/models/df/ner_detached/flat-2021-09-16-15-25-29-f1_98.pth', map_location=self.flat_device_map[intent]),
                # ],
            'Calendar-Query': [
                torch.load('/ai/223/person/lichunyu/models/df/ner_detached/flat-2021-09-19-12-48-41-f1_95.pth', map_location=self.flat_device_map[intent]),
                torch.load('/ai/223/person/lichunyu/models/df/ner_detached/flat-2021-09-19-13-10-00-f1_95.pth', map_location=self.flat_device_map[intent]),
                torch.load('/ai/223/person/lichunyu/models/df/ner_detached/flat-2021-09-19-13-04-26-f1_95.pth', map_location=self.flat_device_map[intent]),
                ]
        }
        inent2train_path = {
            'Alarm-Update': [
                "/ai/223/person/lichunyu/datasets/dataf/seq_label/Alarm-Update_detached_clean.train",
                ],
            'FilmTele-Play': "/ai/223/person/lichunyu/datasets/dataf/seq_label/FilmTele-Play_detached.train",
            # 'Calendar-Query': '/ai/223/person/lichunyu/datasets/dataf/seq_label/Calendar-Query_detached.train',
            'Calendar-Query': [
                '/ai/223/person/lichunyu/datasets/dataf/seq_label/Calendar-Query-872.train',
                '/ai/223/person/lichunyu/datasets/dataf/seq_label/Calendar-Query-872.train',
                '/ai/223/person/lichunyu/datasets/dataf/seq_label/Calendar-Query-872.train',
            ],
            # 'date_and_time': '/ai/223/person/lichunyu/datasets/dataf/seq_label/date_and_time_clean_detached.train'
            # 'date_and_time': [
            #     '/ai/223/person/lichunyu/datasets/dataf/seq_label/date_and_time_detached_clean.train',
            #     '/ai/223/person/lichunyu/datasets/dataf/seq_label/date_and_time_detached_clean.train',
            #     '/ai/223/person/lichunyu/datasets/dataf/seq_label/date_and_time_detached_clean.train',
            # ],
            'date_and_time': '/ai/223/person/lichunyu/datasets/dataf/seq_label/date_and_time_clean_detached.train',
        }
        intent2dev_path = {
            "Alarm-Update": [
                "/ai/223/person/lichunyu/datasets/dataf/seq_label/Alarm-Update_detached_clean.test",
                ],
            'FilmTele-Play': "/ai/223/person/lichunyu/datasets/dataf/seq_label/FilmTele-Play_detached.test",
            'date_and_time': '/ai/223/person/lichunyu/datasets/dataf/seq_label/date_and_time_clean_detached.test',
            # 'date_and_time': [
            #     "/ai/223/person/lichunyu/datasets/dataf/seq_label/date_and_time_detached_clean.test",
            #     "/ai/223/person/lichunyu/datasets/dataf/seq_label/date_and_time_detached_clean.test",
            #     "/ai/223/person/lichunyu/datasets/dataf/seq_label/date_and_time_detached_clean.test",
            #     # "/ai/223/person/lichunyu/datasets/dataf/seq_label/date_and_time_clean_detached.test",
            #     ],
            # 'Calendar-Query': "/ai/223/person/lichunyu/datasets/dataf/seq_label/Calendar-Query_detached.test"
            'Calendar-Query': [
                "/ai/223/person/lichunyu/datasets/dataf/seq_label/Calendar-Query-872.test",
                "/ai/223/person/lichunyu/datasets/dataf/seq_label/Calendar-Query-872.test",
                "/ai/223/person/lichunyu/datasets/dataf/seq_label/Calendar-Query-872.test",
            ]
        }
        intent2test_path = {
            'date_and_time': "/ai/223/person/lichunyu/datasets/dataf/test/test_A_text.nonletter",
            # 'date_and_time': [
            #     "/ai/223/person/lichunyu/datasets/dataf/test/test_A_text.nonletter",
            #     "/ai/223/person/lichunyu/datasets/dataf/test/test_A_text.nonletter",
            #     "/ai/223/person/lichunyu/datasets/dataf/test/test_A_text.nonletter",
            #     ],
            'Alarm-Update': [
                "/ai/223/person/lichunyu/datasets/dataf/test/test_A_text.nonletter",
                ],
            'FilmTele-Play': None,
            'Calendar-Query': [
                "/ai/223/person/lichunyu/datasets/dataf/test/test_A_text.nonletter",
                "/ai/223/person/lichunyu/datasets/dataf/test/test_A_text.nonletter",
                "/ai/223/person/lichunyu/datasets/dataf/test/test_A_text.nonletter",
                ]
        }
        intent2placeholder_path = {
             'date_and_time': "/ai/223/person/lichunyu/datasets/dataf/test/train.nonletter",
            # 'date_and_time': [
            #     "/ai/223/person/lichunyu/datasets/dataf/test/train.nonletter",
            #     # "/ai/223/person/lichunyu/datasets/dataf/test/train.nonletter",
            #     # "/ai/223/person/lichunyu/datasets/dataf/test/train.nonletter",
            #     ],
            'Alarm-Update': [
                "/ai/223/person/lichunyu/datasets/dataf/test/train.nonletter",
                ],
            'FilmTele-Play': None,
            'Calendar-Query': [
                "/ai/223/person/lichunyu/datasets/dataf/test/train.nonletter",
                "/ai/223/person/lichunyu/datasets/dataf/test/train.nonletter",
                "/ai/223/person/lichunyu/datasets/dataf/test/train.nonletter",
            ]
        }
        idx2label_map = {
            'Alarm-Update': I2L_ALARM_UPDATE,
            'FilmTele-Play': I2L_FILMTELE_PLAY,
            'date_and_time': I2L_DATE_AND_TIME,
            'Calendar-Query': I2L_CALENDAR_QUERY,
        }
        res = self._flat_ner_all(intent2model[intent], real_text_list, train_path=inent2train_path[intent], dev_path=intent2dev_path[intent], \
                _device=self.flat_device_map[intent], idx2label=idx2label_map[intent], placeholder_path=intent2placeholder_path[intent], \
                test_path=intent2test_path[intent])
        if manager is not None:
            manager[intent] = res
        return res
        # return self._flat_ner_all(model, real_text_list, train_path=inent2train_path[intent], dev_path=intent2dev_path[intent], \
        #         _device=self.device_map[intent], idx2label=I2L_ALARM_UPDATE)

    def _flat_ner_all(self, _model, real_text_list, train_path=None, dev_path=None, idx2label=None, with_placeholder=True, _device='cuda', \
                        placeholder_path=None, test_path=None):
        slots_list = []
        ensemble = False

        if isinstance(_model, list):
            ensemble = True

        if not isinstance(_model, list):
            _model = [_model]

        if not isinstance(_device, list):
            _device = [_device] * len(_model)

        def create_dataloader(_train_path, _dev_path, _test_path, _placeholder_path, _with_placeholder=False):
            datasets, vocabs, embeddings = load_ner(
                '/root/hub/golden-horse/data',
                '/root/pretrain-models/flat/gigaword_chn.all.a2b.uni.ite50.vec',
                '/root/pretrain-models/flat/gigaword_chn.all.a2b.bi.ite50.vec',
                _refresh=True,
                index_token=False,
                with_placeholder=_with_placeholder,
                train_path=_train_path,
                dev_path=_dev_path,
                test_path=_test_path,
                placeholder_path=_placeholder_path
            )

            w_list = load_yangjie_rich_pretrain_word_list(self.yangjie_rich_pretrain_word_path,
                                                        _refresh=True,
                                                        _cache_fp='cache/{}'.format('yj'))


            datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(datasets,
                                                                        vocabs,
                                                                        embeddings,
                                                                        w_list,
                                                                        self.yangjie_rich_pretrain_word_path,
                                                                        _refresh=True,
                                                                        only_lexicon_in_train=False,
                                                                        word_char_mix_embedding_path=self.yangjie_rich_pretrain_char_and_word_path,
                                                                        number_normalized=0,
                                                                        lattice_min_freq=1,
                                                                        only_train_min_freq=True,
                                                                        with_placeholder=_with_placeholder
                                                                        )

            def collate_func(batch_dict):
                batch_len = len(batch_dict)
                max_seq_length = max([dic['seq_len'] for dic in batch_dict])
                chars = pad_sequence([i['chars'] for i in batch_dict], batch_first=True)
                target = pad_sequence([i['target'] for i in batch_dict], batch_first=True)
                bigrams = pad_sequence([i['bigrams'] for i in batch_dict], batch_first=True)
                seq_len = torch.tensor([i['seq_len'] for i in batch_dict])
                lex_num = torch.tensor([i['lex_num'] for i in batch_dict])
                lex_s = pad_sequence([i['lex_s'] for i in batch_dict], batch_first=True)
                lex_e = pad_sequence([i['lex_e'] for i in batch_dict], batch_first=True)
                lattice = pad_sequence([i['lattice'] for i in batch_dict], batch_first=True)
                pos_s = pad_sequence([i['pos_s'] for i in batch_dict], batch_first=True)
                pos_e = pad_sequence([i['pos_e'] for i in batch_dict], batch_first=True)
                raw_chars = [i['raw_chars'] for i in batch_dict]
                return [chars, target, bigrams, seq_len, lex_num, lex_s, lex_e, lattice, pos_s, pos_e, raw_chars]

            for k, v in datasets.items():
                v.set_input('lattice','bigrams','target', 'seq_len')
                v.set_input('lex_num','pos_s','pos_e')
                v.set_target('target', 'seq_len')
                v.set_pad_val('lattice',vocabs['lattice'].padding_idx)


            dev_ds = NERDataset(datasets['test'])
            dev_dataloader = DataLoader(
                dev_ds,
                batch_size=1,
                collate_fn=collate_func
            )
            return dev_dataloader

        dataloader = []

        if ensemble is True:
            for idx in range(len(_model)):
                if isinstance(with_placeholder, bool):
                    with_placeholder = [with_placeholder] * len(_model)
                dataloader_item = create_dataloader(train_path[idx], dev_path[idx], test_path[idx], placeholder_path[idx], with_placeholder[idx])
                dataloader.append(dataloader_item)
        else:
            dataloader_item = create_dataloader(train_path, dev_path, test_path, placeholder_path, with_placeholder)
            dataloader.append(dataloader_item)

        for m in _model:
            m.eval()

        dataloader = [list(_) for _ in dataloader]

        for step in range(len(dataloader[0])):
            #TODO BERT embedding 前20 epoch 冻结

            trans_m, mask, logits = [], [], []

            for idx in range(len(_model)):
                batch = dataloader[idx][step]
                real_text = real_text_list[step]
                target = batch[1].to(_device[idx])
                bigrams = batch[2].to(_device[idx])
                seq_len = batch[3].to(_device[idx])
                lex_num = batch[4].to(_device[idx])
                lattice = batch[7].to(_device[idx])
                pos_s = batch[8].to(_device[idx])
                pos_e = batch[9].to(_device[idx])
                _, offsets = text_rm_space(real_text)
                text_list = [real_text]

                with torch.no_grad():

                    output = _model[idx](
                        lattice,
                        bigrams,
                        seq_len,
                        lex_num,
                        pos_s,
                        pos_e,
                        target
                    )
                # pred = output['pred']
                _trans_m = output['trans_m']
                _mask = output['mask']
                _logits = output['logits']
                trans_m.append(_trans_m)
                mask.append(_mask)
                logits.append(_logits)
            pred_ensemble, score = viterbi_ensemble_decode(logits=logits, mask=mask, trans_m=trans_m)
            res = ner_extract(pred_ensemble, seq_len, text_list, idx2label=idx2label, offsets=offsets)
            slots_list.append(flat_slot_clean(res))
        return slots_list


    def go(self, data_path, output_path, with_text:bool=False, full_slot:bool=False):
        bt = time.time()
        res = defaultdict(dict)
        test_data = json.load(open(data_path, 'r'))
        if self.policy['ner'] == 'flat':
            flat_text_list = []
            for idx, text_dict in test_data.items():
                text = text_dict['text']
                flat_text_list.append(text)
            
            # _ = self.flat_ner_all(flat_text_list, 'all')
            torch.multiprocessing.set_start_method('spawn')
            manager = torch.multiprocessing.Manager()
            manager = manager.dict()
            p_slots_alarm_update = torch.multiprocessing.Process(target=self.flat_ner_all, args=(flat_text_list, 'Alarm-Update', manager))
            p_slots_date_and_time = torch.multiprocessing.Process(target=self.flat_ner_all, args=(flat_text_list, 'date_and_time', manager))
            p_slots_all = torch.multiprocessing.Process(target=self.flat_ner_all, args=(flat_text_list, 'all', manager))
            p_slots_calendar_query = torch.multiprocessing.Process(target=self.flat_ner_all, args=(flat_text_list, 'Calendar-Query', manager))
            jobs = [p_slots_alarm_update, p_slots_all, p_slots_date_and_time, p_slots_calendar_query]
            for p in jobs:
                p.start()

            for p in jobs:
                p.join()

            slots_list_by_alarm_update = manager['Alarm-Update']
            # slots_list_by_filmtele_play = manager['FilmTele-Play']
            slots_list_date_and_time = manager['date_and_time']
            slots_list_calendar_query = manager['Calendar-Query']
            slots_list_all = manager['all']
            slots_candidate = {
                'Music-Play': slots_list_all,
                'HomeAppliance-Control': slots_list_all,
                'Travel-Query': slots_list_all,
                'Calendar-Query': slots_list_calendar_query,
                'FilmTele-Play': slots_list_all,
                'Weather-Query': slots_list_all,
                'Video-Play': slots_list_all,
                'Alarm-Update': slots_list_by_alarm_update,
                'TVProgram-Play': slots_list_all,
                'Audio-Play': slots_list_all,
                'Radio-Listen': slots_list_all
            }
        for n_idx, (idx, text_dict) in tqdm(enumerate(test_data.items())):
            text = text_dict['text']
            ood = self.is_ood(text)
            if ood is True:
                intent = INTENT[-1]
            else:
                intent = self.classify(text)
            res[idx]['intent'] = intent
            if self.policy['only_cls'] is True:
                res[idx]['slots'] = {}
                continue
            if intent == 'Other':
                res[idx]['slots'] = {}
                continue
            if self.policy['ner'] == 'flat':
                slots = slots_candidate[intent][n_idx]
                if intent not in ['Calendar-Query']:
                # if intent not in ['1']:
                    date_all, time_all = '', ''
                    if 'datetime_date' in slots:
                        date_all = slots.pop('datetime_date')
                    if 'datetime_time' in slots:
                        time_all = slots.pop('datetime_time')
                    slots_detached_date_and_time = slots_list_date_and_time[n_idx]

                    # date and time ensemble
                    if 'datetime_date' in slots_detached_date_and_time:
                        if isinstance(slots_detached_date_and_time['datetime_date'], list):
                            tmp = []
                            for i in slots_detached_date_and_time['datetime_date']:
                                if len(i) != 1:
                                    tmp.append(i)
                            if not tmp:
                                slots_detached_date_and_time['datetime_date'] = ''
                            else:
                                slots_detached_date_and_time['datetime_date'] = list(set(tmp)) if len(list(set(tmp))) > 1 else list(set(tmp))[0]
                        if isinstance(slots_detached_date_and_time['datetime_date'], str):
                            if len(slots_detached_date_and_time['datetime_date']) == 1:
                                if len(date_all) not in [0, 1]:
                                    slots_detached_date_and_time['datetime_date'] = date_all
                                else:
                                    del slots_detached_date_and_time['datetime_date']

                    if 'datetime_time' in slots_detached_date_and_time:
                        if isinstance(slots_detached_date_and_time['datetime_time'], list):
                            tmp = []
                            for i in slots_detached_date_and_time['datetime_time']:
                                if len(i) != 1:
                                    tmp.append(i)
                            if not tmp:
                                slots_detached_date_and_time['datetime_time'] = ''
                            else:
                                slots_detached_date_and_time['datetime_time'] = list(set(tmp)) if len(list(set(tmp))) > 1 else list(set(tmp))[0]
                        if isinstance(slots_detached_date_and_time['datetime_time'], str):
                            if len(slots_detached_date_and_time['datetime_time']) == 1:
                                if len(time_all) not in [0, 1]:
                                    slots_detached_date_and_time['datetime_time'] = time_all
                                else:
                                    del slots_detached_date_and_time['datetime_time']


                    slots.update(slots_list_date_and_time[n_idx])
                slots = clean_slot_by_intent(slots, intent, post=False)
                if intent == 'Alarm-Update':
                    if 'notes' in slots:
                        if ''.join(slots['notes']) in text:
                            slots['notes'] = ''.join(slots['notes'])
                        elif 'notes' in slots_list_all[n_idx] and isinstance(slots_list_all[n_idx]['notes'], str) and isinstance(slots['notes'], list):
                            _mark = True
                            for _ in slots['notes']:
                                if _ not in slots_list_all[n_idx]['notes']:
                                    _mark = False
                                    break
                            if _mark is True:
                                slots['notes'] = slots_list_all[n_idx]['notes']

            else:
                slots = self.ner(text, intent)
                slots = clean_slot_by_intent(slots, intent, post=False)

            if intent == 'Music-Play':
                play_mode = self.music_play_play_mode(text)
                if play_mode:
                    slots['play_mode'] = play_mode
                language = self.music_play_language(text)
                if language:
                    slots['language'] = language

            if intent == 'Audio-Play':
                language = self.music_play_language(text)
                if language:
                    slots['language'] = language

            if intent == 'Weather-Query':
                index = self.weather_query_index(text)
                if index:
                    slots['index'] = index

                _type = self.weather_query_type(text)
                if _type:
                    slots['type'] = _type

            if intent == 'Travel-Query':
                query_type = self.travel_query_query_type(text)
                if query_type:
                    slots['query_type'] = query_type

            if intent == 'HomeAppliance-Control':
                command = self.home_appliance_control_command(text)
                if command:
                    slots['command'] = command

            if intent == 'FilmTele-Play' or intent == 'Video-Play':
                region = self.region_extract(text)
                if region:
                    slots['region'] = region

            slots = slot_hash(slots)
            time_patch_slot = weather_time_patch(text)
            if time_patch_slot['datetime_time']:
                if intent == 'Weather-Query':
                    slots = slot_add(slots, time_patch_slot)
                else:
                    slots = slot_delete(slots, time_patch_slot)

            slots = slot_delete(slots, {'datetime_time': ['一会儿', '一会']})

            if with_text is True:
                res[idx]['text'] = text

            if full_slot is True:
                for co_slot in NER_CORRESPONDENCE[intent]:
                    if co_slot not in slots:
                        slots[co_slot] = ''

            res[idx]['slots'] = slots
            if slots == {} and intent == 'Calendar-Query':
                res[idx]['intent'] = 'Other'

            if 'name' in slots and isinstance(slots['name'], list):
                if ''.join(slots['name']) in text:
                    slots['name'] = ''.join(slots['name'])

        with open(output_path, 'w') as f:
            res_jsoned = json.dumps(res, ensure_ascii=False, indent=4)
            f.write(res_jsoned)
        print('total cost {}s'.format(time.time()-bt))
        return output_path


    def trash_flat_ner_item(self, text, intent):
        """单条flat抽取，暂时弃用

        :param text: [description]
        :type text: [type]
        :param intent: [description]
        :type intent: [type]
        :return: [description]
        :rtype: [type]
        """

        tmp_test_path = '/ai/223/person/lichunyu/datasets/tmp/test_one.txt'

        with open(tmp_test_path, 'w') as f:
            normalization_text = normalization(text, letter='@')
            normalization_text = normalization_text.replace(' ', '')
            for char in normalization_text:
                f.write(char + '\t' + 'O' + '\n')


        datasets, vocabs, embeddings = load_ner(
            '/root/hub/golden-horse/data',
            '/root/pretrain-models/flat/gigaword_chn.all.a2b.uni.ite50.vec',
            '/root/pretrain-models/flat/gigaword_chn.all.a2b.bi.ite50.vec',
            _refresh=True,
            index_token=False,
            test_path=tmp_test_path,
        )

        w_list = load_yangjie_rich_pretrain_word_list(self.yangjie_rich_pretrain_word_path,
                                                    _refresh=True,
                                                    _cache_fp='cache/{}'.format('yj'))


        datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(datasets,
                                                                    vocabs,
                                                                    embeddings,
                                                                    w_list,
                                                                    self.yangjie_rich_pretrain_word_path,
                                                                    _refresh=True,
                                                                    # _cache_fp=cache_name,
                                                                    only_lexicon_in_train=False,
                                                                    word_char_mix_embedding_path=self.yangjie_rich_pretrain_char_and_word_path,
                                                                    number_normalized=0,
                                                                    lattice_min_freq=1,
                                                                    only_train_min_freq=True
                                                                    )

        def collate_func(batch_dict):
            batch_len = len(batch_dict)
            max_seq_length = max([dic['seq_len'] for dic in batch_dict])
            chars = pad_sequence([i['chars'] for i in batch_dict], batch_first=True)
            target = pad_sequence([i['target'] for i in batch_dict], batch_first=True)
            bigrams = pad_sequence([i['bigrams'] for i in batch_dict], batch_first=True)
            seq_len = torch.tensor([i['seq_len'] for i in batch_dict])
            lex_num = torch.tensor([i['lex_num'] for i in batch_dict])
            lex_s = pad_sequence([i['lex_s'] for i in batch_dict], batch_first=True)
            lex_e = pad_sequence([i['lex_e'] for i in batch_dict], batch_first=True)
            lattice = pad_sequence([i['lattice'] for i in batch_dict], batch_first=True)
            pos_s = pad_sequence([i['pos_s'] for i in batch_dict], batch_first=True)
            pos_e = pad_sequence([i['pos_e'] for i in batch_dict], batch_first=True)
            raw_chars = [i['raw_chars'] for i in batch_dict]
            return [chars, target, bigrams, seq_len, lex_num, lex_s, lex_e, lattice, pos_s, pos_e, raw_chars]

        for k, v in datasets.items():
            v.set_input('lattice','bigrams','target', 'seq_len')
            v.set_input('lex_num','pos_s','pos_e')
            v.set_target('target', 'seq_len')
            v.set_pad_val('lattice',vocabs['lattice'].padding_idx)


        dev_ds = NERDataset(datasets['test'])
        dev_dataloader = DataLoader(
            dev_ds,
            batch_size=1,
            collate_fn=collate_func
        )

        self.ner_model.eval()
        for step, batch in enumerate(dev_dataloader):
            #TODO BERT embedding 前20 epoch 冻结
            # chars = batch[0].cuda()
            target = batch[1].cuda()
            bigrams = batch[2].cuda()
            seq_len = batch[3].cuda()
            lex_num = batch[4].cuda()
            # lex_s = batch[5].cuda()
            # lex_e = batch[6].cuda()
            lattice = batch[7].cuda()
            pos_s = batch[8].cuda()
            pos_e = batch[9].cuda()
            raw_chars = batch[10]
            text_list = [''.join(i) for i in raw_chars]

            with torch.no_grad():

                output = self.ner_model(
                    lattice,
                    bigrams,
                    seq_len,
                    lex_num,
                    pos_s,
                    pos_e,
                    target
                )
            pred = output['pred']
            res = ner_extract(pred, seq_len, text_list, idx2label=IDX2LABEL)
            return flat_slot_clean(res, intent=intent, with_intent=True)


def slot_add(slot_origin, slot_new):
    for k, v in slot_new.items():
        if k not in slot_origin:
            slot_origin[k] = v
        else:
            if isinstance(slot_origin[k], str):
                slot_origin[k] = [slot_origin[k]]
            if isinstance(slot_new[k], list):
                slot_origin[k].extend(slot_new[k])
            else:
                slot_origin[k].append(slot_new[k])
            slot_origin[k] = list(set(slot_origin[k]))
            if len(slot_origin[k]) == 1:
                slot_origin[k] = slot_origin[k][0]
    return slot_origin


def slot_delete(slot_origin, slot_new):
    for k, v in slot_new.items():
        if k in slot_origin:
            if isinstance(slot_new[k], str):
                slot_new[k] = [slot_new[k]]
            for i in slot_new[k]:
                if k not in slot_origin:
                    break
                if isinstance(slot_origin[k], str):
                    if slot_origin[k] == i:
                        del slot_origin[k]
                else:
                    if i in slot_origin[k]:
                        slot_origin[k].remove(i)

            if k in slot_origin and isinstance(slot_origin[k], list):
                slot_origin[k] = list(set(slot_origin[k]))
                if len(slot_origin[k]) == 1:
                    slot_origin[k] = slot_origin[k][0]
                elif len(slot_origin[k]) == 0:
                    del slot_origin[k]

    return slot_origin


def weather_time_patch(text):
    db = [
        # ['一会儿', '一会'],
        '现在'
    ]
    res = {
        'datetime_time': []
    }
    for d in db:
        if isinstance(d, list):
            for i in d:
                if i in text:
                    res['datetime_time'].append(i)
                    break
        else:
            if d in text:
                res['datetime_time'].append(d)

    if len(res['datetime_time']) == 0:
        return {'datetime_time': ''}
    if len(res['datetime_time']) == 1:
        return {'datetime_time': res['datetime_time'][0]}
    return res


def slot_hash(slot):
    if 'type' in slot and 'index' in slot:
        if slot['type'] == '雨' and slot['index'] == '降水量':
            return {k:v for k,v in slot.items() if k != 'type'}
    return slot


def dict_extract_one_value(dic):
    for k, v in dic.items():
        return v

    return {}


def flat_slot_clean(slots_dict, intent=None, with_intent=False):
    slots_dict = dict_extract_one_value(slots_dict)
    res = {}
    for slot_k, slot_v in slots_dict.items():
        real_slot_k = slot_k
        for i in INTENT:
            real_slot_k = real_slot_k.replace(i+'-', '')
        if len(slot_v) == 1:
            res[real_slot_k] = slot_v[0]
        else:
            res[real_slot_k] = slot_v
    return res


def clean_slot_by_intent(slots_dict, intent, post=False):
    res = {}
    for k, v in slots_dict.items():
        if post is True:
            if intent not in k:
                continue
            k = k.replace(intent+'-', '')
        if k not in CORRESPONDENCE[intent]:
            # print('bug found+++++++++++++++++++++++')
            continue
        if not v:
            continue
        res[k] = v
    return res


def post_process(intent, slots):
    pass


def normalization(text, letter='@'):
    candidate_num_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    candidate_letter_list = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
        'v', 'w', 'x', 'y', 'z',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
        'V', 'W', 'X', 'Y', 'Z'
    ]

    for i in candidate_num_list:
        text = text.replace(i, '*')

    for i in candidate_letter_list:
        text = text.replace(i, letter)

    return text


if __name__ == '__main__':
    nlpgogo = NlpGoGo(
        cls_model_path='/ai/223/person/lichunyu/models/df/best/bert-2021-08-05-03-19-23-f1_99.pth',  # few_shot
        cls_ext_model_path='/ai/223/person/lichunyu/models/df/best/bert-2021-09-02-12-19-11-f1_99.pth', # hfl ext eopch 3
        ood_model_path='/ai/223/person/lichunyu/models/df/best/bert-2021-08-13-02-54-15-f1_98.pth',
        ood_bert_model_path='/ai/223/person/lichunyu/models/df/best/bert-2021-08-13-02-54-15-f1_98.pth',
        ood_bert_6000_model_path='/ai/223/person/lichunyu/models/df/best/bert-2021-08-23-04-01-14-f1_99.pth',
        # ood_roberta_model_path='/ai/223/person/lichunyu/models/df/best/bert-2021-08-19-04-09-06-f1_96.pth',  # roberta
        # ood_roberta_6000_model_path='/ai/223/person/lichunyu/models/df/best/bert-2021-08-23-07-32-11-f1_96.pth', # roberta
        ood_fake_roberta_6000_model_path='/ai/223/person/lichunyu/models/df/best/bert-2021-08-23-08-38-13-f1_98.pth',
        ood_fake_roberta_2700_model_path='/ai/223/person/lichunyu/models/df/best/bert-2021-08-23-09-28-31-f1_98.pth',
        ood_macbert_model_path='/ai/223/person/lichunyu/models/df/best/bert-2021-08-24-08-49-40-f1_98.pth',
        ood_hfl_model_path='/ai/223/person/lichunyu/models/df/best/bert-2021-09-06-14-47-38-f1_98.pth',
        cls_config_path='/root/pretrain-models/bert-base-chinese',
        ner_model_path='/ai/223/person/lichunyu/models/df/best/bert-2021-09-01-10-31-48-f1_88.pth',
        policy={
            'cls_model': 'ensemble',
            'only_cls': False,
            'ner': 'flat',
            'ood_model': 'ensemble'
        },
        flat_device_map = {
            'all': 'cuda:1',
            'Alarm-Update': 'cuda:6',
            'FilmTele-Play': 'cuda:1',
            'date_and_time': 'cuda:2',
            'Calendar-Query': 'cuda:3'
        },
        ood_device_map = {
            'bert': 'cuda:0'
        },
        cls_device_map = {
            'bert': 'cuda:2'
        },
        # query_type_model_path='/ai/223/person/lichunyu/models/df/query_type/bert-2021-07-29-02-27-35-f1_97.pth',
        query_type_model_path='/ai/223/person/lichunyu/models/df/best/bert-2021-07-29-07-48-11-f1_96.pth',
        # command_model_path='/ai/223/person/lichunyu/models/df/command/bert-2021-07-29-02-34-54-f1_97.pth'
        command_model_path='/ai/223/person/lichunyu/models/df/best/bert-2021-07-29-07-43-15-f1_97.pth',
        ood_config_path='/root/pretrain-models/hfl-chinese-roberta-wwm-ext-large',
        # flat_all_model_path='/ai/223/person/lichunyu/models/df/best/flat-2021-08-30-22-03-15-f1_92.pth',
        flat_all_model_path='/ai/223/person/lichunyu/models/df/ner_detached/flat-2021-09-13-19-37-25-f1_91.pth',
        flat_alarm_update_model_path=[
            '/ai/223/person/lichunyu/models/df/ner/flat-2021-09-10-15-03-03-f1_97.pth',
            '/ai/223/person/lichunyu/models/df/ner/flat-2021-09-10-14-41-33-f1_95.pth',
            '/ai/223/person/lichunyu/models/df/ner/flat-2021-09-10-14-53-13-f1_95.pth',
            '/ai/223/person/lichunyu/models/df/ner/flat-2021-09-10-14-38-33-f1_95.pth',
        ],
        flat_filmtele_play_model_path='/ai/223/person/lichunyu/models/df/best/flat-2021-09-06-15-16-14-f1_94.pth'
    )

    _ = nlpgogo.go(
        data_path='/ai/223/person/lichunyu/datasets/dataf/test/test_A_text.json',
        # data_path='/root/train.json',  # Music-Play
        output_path='/ai/223/person/lichunyu/datasets/dataf/output/output.json'
        # output_path='/ai/223/person/lichunyu/datasets/dataf/output/Radio-Listen.json',
        # with_text=True,
        # full_slot=True 
    )
