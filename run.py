# -*- encoding: utf-8 -*-
'''
@create_time: 2021/08/31 10:32:21
@author: lichunyu
'''

from run.run_chinese_ner import main as run_chinese_ner
from run.run_classification import main as run_classification




if __name__ == '__main__':
    run_classification('/root/EarleeNLP/args/df_intent_ood_macbert.json')