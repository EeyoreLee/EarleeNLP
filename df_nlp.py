# -*- encoding: utf-8 -*-
'''
@create_time: 2021/07/23 15:22:53
@author: lichunyu
'''

import json
from collections import defaultdict

from transformers import (
    BertTokenizer
)
import torch
from tqdm import tqdm
import torch.nn.functional as F


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


# QUERY_MAP = {
#     'name': '用以识别某一个体或群体(人和事物)的专属名词',
#     'datetime_date': '发生某一事情的确定的日子或时期',
#     'departure': '出发的地方',
#     'instrument': '用来演奏的乐器',
#     'datetime_time': '物质的永恒运动、变化的持续性、顺序性的表现',
#     'destination': '想要达到的地方',
#     'appliance': '工作时所需用的器具',
#     'notes': '听课、听报告、读书时所做的记录',
#     'details': '所完成的具体的事或行动',
#     'play_setting': '播放的方式',
#     'artist': '杂技、戏曲、民间歌舞、曲艺演员',
#     'city': '人口集中，居民以非农业人口为主，工商业比较发达的地区',
#     'frequency': '每个对象出现的次数与总次数的比值',
#     'channel': '电视台或电视网络',
#     'age': '具体的年份日期',
#     'album': '歌曲集合或专辑',
#     'tag': '具有相同特征的事物所形成的类别',
#     'song': '用来歌唱的音乐或歌曲',
#     'region': '某一范围的地方'
# }


SUB_CLS_LABEL = [
    'query_type',
    'play_mode',
    'command',
    'index',
    'language',
    # 'region',
    'type'
]


NER_CORRESPONDENCE = {
    'Travel-Query': ['datetime_date', 'departure', 'datetime_time', 'destination'],
    'Music-Play': ['instrument', 'artist', 'age', 'album', 'song'],
    'FilmTele-Play': ['name', 'artist', 'play_setting', 'age', 'region', 'tag'],
    'Video-Play': ['name', 'region', 'datetime_time', 'datetime_date'],
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


class NlpGoGo(object):

    def __init__(self, cls_model_path, cls_config_path, ner_model_path, ood_model_path, query_type_model_path=None, command_model_path=None, \
                 device='cuda', max_length=150, policy:dict=None):
        """
        policy = {
                'cls_model': 'bert',
                'only_cls': False,
                    }
        """
        self.device = device
        self.policy = policy
        if policy['cls_model'] == 'bert':
            self.cls_model = torch.load(cls_model_path, map_location=torch.device(device))
            self.tokenizer = BertTokenizer.from_pretrained(cls_config_path)
            self.max_length = max_length

        if policy['only_cls'] is False:
            m = torch.load(ner_model_path, map_location=torch.device(device))
            self.ner_model = m

        self.odd_model = torch.load(ood_model_path, map_location=torch.device(device))

        self.query_type_model = torch.load(query_type_model_path, map_location=torch.device(device))
        self.command_model = torch.load(command_model_path, map_location=torch.device(device))

    def is_ood(self, text) -> bool:
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

        self.odd_model.eval()
        with torch.no_grad():
            input_ids = input_ids.to(self.device).to(torch.int64)
            attention_mask = attention_mask.to(self.device).to(torch.int64)
            output = self.odd_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        id2label = {idx: label for idx, label in enumerate(INTENT)}
        if torch.max(F.softmax(output.logits)).detach().cpu().numpy().tolist() < 0.95:
            intent = INTENT[-1]
        else:
            intent = id2label[torch.argmax(output.logits).detach().cpu().numpy().tolist()]

        if intent != INTENT[-1]:
            return False
        return True

    def classify(self, text):
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

        self.cls_model.eval()
        with torch.no_grad():
            input_ids = input_ids.to(self.device).to(torch.int64)
            attention_mask = attention_mask.to(self.device).to(torch.int64)
            output = self.cls_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        id2label = {idx: label for idx, label in enumerate(INTENT)}
        # if torch.max(F.softmax(output.logits)).detach().cpu().numpy().tolist() < 0.95:
        #     return INTENT[-1]
        return id2label[torch.argmax(output.logits).detach().cpu().numpy().tolist()]

    def ner(self, text, classification):
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
    #     if '循环播放' in text:
    #         return '循环播放'
        cls_2 = [
            '顺序播放',
            '挨着',
            '挨个'
        ]
        for i in cls_2:
            if i in text:
                return '顺序播放'
        cls_0 = [
            '播放一下',
            '随机',
            '随便',
            '来一首',
            '找一首',
            '放一首',
            '听一下',
            '听一首',
            '任意',
            '来首',
            '来一个'
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
            '降水',
            '降雨'
        ]
        for i in cls_5:
            if i in text:
                return '降水量'
        cls_0 = [
            '穿',
            '多加衣物'
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
        if '风' in text:
            return '风力'
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
        if '衣物' in text:
            return '穿衣指数'
        if '暴雨' in text:
            return '暴雨'
        if '雨' in text:
            return '雨'
        if '晴' in text:
            return '晴'
        if '暴雪' in text:
            return '暴雪'
        if '台风' in text:
            return '台风'
        if '风' in text:
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

    def go(self, data_path, output_path):
        res = defaultdict(dict)
        test_data = json.load(open(data_path, 'r'))
        for idx, text_dict in tqdm(test_data.items()):
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
            slots = self.ner(text, intent)

            if intent == 'Music-Play':
                play_mode = self.music_play_play_mode(text)
                if play_mode:
                    slots['play_mode'] = play_mode
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

            res[idx]['slots'] = slots

        with open(output_path, 'w') as f:
            res_jsoned = json.dumps(res, ensure_ascii=False)
            f.write(res_jsoned)

        return output_path


def  normalization(text):
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
        text = text.replace(i, '#')

    return text


if __name__ == '__main__':
    nlpgogo = NlpGoGo(
        cls_model_path='/ai/223/person/lichunyu/models/tmp/bert-2021-07-23-07-21-34-f1_98.pth',
        ood_model_path='/ai/223/person/lichunyu/models/df/intent/bert-2021-08-02-14-35-46-f1_99.pth',  # odd acc 0.95
        # cls_model_path='/ai/223/person/lichunyu/models/tmp/bert-2021-07-29-07-24-33-f1_97.pth',
        cls_config_path='/root/pretrain-models/bert-base-chinese',
        ner_model_path='/ai/223/person/lichunyu/models/tmp/bert-2021-07-28-15-18-42-f1_64.pth',
        # ner_model_path='/ai/223/person/lichunyu/models/df/mrc-ner/bert-2021-08-02-07-30-50-f1_60.pth',  
        policy={
            'cls_model': 'bert',
            'only_cls': False
        },
        # query_type_model_path='/ai/223/person/lichunyu/models/df/query_type/bert-2021-07-29-02-27-35-f1_97.pth',
        query_type_model_path='/ai/223/person/lichunyu/models/df/query_type/bert-2021-07-29-07-48-11-f1_96.pth',
        # command_model_path='/ai/223/person/lichunyu/models/df/command/bert-2021-07-29-02-34-54-f1_97.pth'
        command_model_path='/ai/223/person/lichunyu/models/df/command/bert-2021-07-29-07-43-15-f1_97.pth'
    )

    _ = nlpgogo.go(
        data_path='/ai/223/person/lichunyu/datasets/dataf/test/test_A_text.json',
        # data_path='/root/EarleeNLP/data/datafountain/train.json',
        output_path='/ai/223/person/lichunyu/datasets/dataf/output/output.json'
    )