# -*- encoding: utf-8 -*-
'''
@create_time: 2022/06/24 10:24:59
@author: lichunyu
'''
import logging
import sys
import os
import importlib

import torch
from transformers import HfArgumentParser, TrainingArguments

from utils.generic import AdvanceArguments
from utils.format_util import snake2upper_camel
from utils.trainer import Trainer


def main(json_path=None):
    """An entry point for training any model
    :usage: 1. python main.py {json_path}
            2. python main.py --model flat --output_dir output/flat/
            3. main(json_path)

    :param json_path: Easy to use for debugging, defaults to None
    :type json_path: str
    """    
    parser = HfArgumentParser(AdvanceArguments)
    if json_path:
        advance_args, = parser.parse_json_file(json_file=json_path)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        advance_args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        advance_args, = parser.parse_args_into_dataclasses()

    os.environ["CUDA_VISIBLE_DEVICES"] = advance_args.cuda_visible_devices

    model_segment_name = advance_args.model
    try:
        model_module = importlib.import_module(".".join([
            "models",
            f"{model_segment_name}",
            f"modeling_{model_segment_name}",
        ]))
        model = getattr(model_module, f"{snake2upper_camel(model_segment_name)}")
    except ModuleNotFoundError as e:
        raise ImportError(
            f"no model named {model_segment_name}, please select one from "
            f"{chr(10).join(['']+['* '+i for i in os.listdir('./models') if os.path.isdir(os.path.join('./models', i)) and not i.startswith('_')])}"
        ) from e
    except AttributeError as e:
        raise NotImplementedError(
            f"{snake2upper_camel(model_segment_name)} is not exist. please check it in "
            f"{'/'.join(['./models', model_segment_name, 'modeling_'+model_segment_name])}"
        ) from e

    try:
        init_func = getattr(model_module, f"init_func")
    except AttributeError as e:
        raise NotImplementedError(
            f"init_func is not exist. please check it in "
            f"{'/'.join(['./models', model_segment_name, 'modeling_'+model_segment_name])}"
        ) from e

    try:
        argument_module = importlib.import_module(".".join([
            "models",
            f"{model_segment_name}",
            f"argument_{model_segment_name}",
        ]))
        argument = getattr(argument_module, "ModelArgument")
    except AttributeError as e:
        raise NotImplementedError(
            f"ModelArgument is not exist. please check it in "
            f"{'/'.join(['./models', model_segment_name, 'argument_'+model_segment_name])}"
        ) from e

    try:
        collection_module = importlib.import_module(".".join([
            "models",
            f"{model_segment_name}",
            f"collection_{model_segment_name}",
        ]))
        collection = getattr(collection_module, "Collection")
    except AttributeError as e:
        raise NotImplementedError(
            f"Collection is not exist. please check it in "
            f"{'/'.join(['./models', model_segment_name, 'collection_'+model_segment_name])}"
        ) from e

    parser = HfArgumentParser(argument, TrainingArguments)
    if json_path:
        model_args, training_args = parser.parse_json_file(json_file=json_path)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    model = init_func(model, **model_args.init_param)

    collection = collection(data_path=advance_args.data_path)
    train_dataloader, dev_dataloader = collection()

    trainer = Trainer(
        model=model,
        args=training_args,
        extra_args=[model_args],
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader
    )
    trainer.fit()
    ...



if __name__ == "__main__":
    main("args/refactor_test.json")