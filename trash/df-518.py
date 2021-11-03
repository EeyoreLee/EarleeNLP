# -*- encoding: utf-8 -*-
'''
@create_time: 2021/11/01 16:01:46
@author: lichunyu
'''

from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
from tqdm import tqdm

from deploy.bert_modeling import bert_classification_inference


class DF518(object):

    def __init__(self) -> None:
        super().__init__()
        self.tokenzier = BertTokenizer.from_pretrained('/ai/223/person/lichunyu/pretrain-models/bert-base-chinese')
        # self.model_1 = torch.load('/ai/223/person/lichunyu/models/df-competition/df-518/train_1/bert-2021-11-01-15-09-45-f1_65.pth', map_location='cuda')
        # self.model_2 = torch.load('/ai/223/person/lichunyu/models/df-competition/df-518/train_2/bert-2021-11-01-15-40-58-f1_76.pth', map_location='cuda')
        # self.model_3 = torch.load('/ai/223/person/lichunyu/models/df-competition/df-518/train_3/bert-2021-11-01-15-45-04-f1_76.pth', map_location='cuda')
        # self.model_4 = torch.load('/ai/223/person/lichunyu/models/df-competition/df-518/train_4/bert-2021-11-01-15-54-30-f1_64.pth', map_location='cuda')
        # self.model_5 = torch.load('/ai/223/person/lichunyu/models/df-competition/df-518/train_5/bert-2021-11-01-15-52-41-f1_64.pth', map_location='cuda')
        # self.model_6 = torch.load('/ai/223/person/lichunyu/models/df-competition/df-518/train_6/bert-2021-11-01-15-57-47-f1_64.pth', map_location='cuda')
        self.model = torch.load('/ai/223/person/lichunyu/models/df-competition/df-518/total/bert-2021-11-02-10-19-13-f1_61.pth', map_location='cuda')
        self.test_data_path = '/ai/223/person/lichunyu/datasets/df-competition/df-518/origin/test_dataset.tsv'

    def predict(self):
        df_test = pd.read_csv(self.test_data_path, sep='\t')
        result = {
            'id': [],
            'emotion': []
        }
        # model_list = [self.model_1, self.model_2, self.model_3, self.model_4, self.model_5, self.model_6]
        for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
            result['id'].append(row['id'])
            tmp_emotion = [0,0,0,0,0,0]
            if isinstance(row['character'], str):
                # text = row['content'].replace(row['character'], '')
                text = row['content'] + ',' + ','.join(5*[row['character']])
                # for m in model_list:
                res_item = bert_classification_inference(text, self.model, self.tokenzier, max_length=220)
                if res_item > 0:
                    tmp_emotion[res_item] = 1
                # tmp_emotion.append(str(res_item))
                tmp_emotion = [str(i) for i in tmp_emotion]
                result['emotion'].append(','.join(tmp_emotion))
            else:
                result['emotion'].append('0,0,0,0,0,0')

        df_predict = pd.DataFrame(result)
        df_predict.to_csv('/ai/223/person/lichunyu/datasets/df-competition/df-518/test/result.csv', index=None, sep='\t')



if __name__ == '__main__':
    df518 = DF518()
    df518.predict()