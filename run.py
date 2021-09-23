# -*- encoding: utf-8 -*-
'''
@create_time: 2021/08/31 10:32:21
@author: lichunyu
'''
import sys
import os

from transformers import HfArgumentParser

from utils.args import TaskArguments
from run.run_chinese_ner import main as run_chinese_ner
from run.run_classification import main as run_classification
from run.run_ner import main as run_ner


def run(json_path=None):
    parser = HfArgumentParser(TaskArguments)
    if json_path:
        task_args, = parser.parse_json_file(json_file=json_path)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        task_args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        task_args, = parser.parse_args_into_dataclasses()

    os.environ['CUDA_VISIBLE_DEVICES'] = task_args.cus_cuda_visible_devices
    task = task_args.task
    if task == 'chinese_ner':
        run_chinese_ner(json_path)
    elif task == 'classification':
        run_classification(json_path)





if __name__ == '__main__':
    run('/root/EarleeNLP/args/with_label/df_all_train_aug3_5flod_1.json')
    # run('/root/EarleeNLP/args/df_alarm_update_ner_aug5.json')
    # run_classification('/root/EarleeNLP/args/df_intent_all_aug.json')
    # run()
