# -*- encoding: utf-8 -*-
'''
@create_time: 2022/07/01 10:21:15
@author: lichunyu
'''

from transformers import TrainingArguments, HfArgumentParser
from utils.generic import AdvanceArguments, get_args


json_path = "/root/workspace/EarleeNLP/args/refactor_test.json"

parser = HfArgumentParser((TrainingArguments, AdvanceArguments))
training_args, adv_args = parser.parse_json_file(json_file=json_path)
model_name = get_args("model", training_args, [adv_args])
...