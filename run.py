# -*- encoding: utf-8 -*-
'''
@create_time: 2021/08/31 10:32:21
@author: lichunyu
'''

from run.run_chinese_ner import main as run_chinese_ner
from run.run_classification import main as run_classification
from run.run_ner import main as run_ner


def run(json_file_path=None):
    pass





if __name__ == '__main__':
    # run_ner('/root/EarleeNLP/args/ner.json')
    # run_chinese_ner('/root/EarleeNLP/args/df_flat.json')
    run_classification('/root/EarleeNLP/args/df_intent_all_aug.json')