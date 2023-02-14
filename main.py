# -*- encoding: utf-8 -*-
'''
@create_time: 2022/06/24 10:24:59
@author: lichunyu
'''
import logging
import sys
import os
import importlib

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    default_data_collator
)

from utils.generic import AdvanceArguments
from utils.format_util import snake2upper_camel
# from utils.trainer import Trainer


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
        advance_args, = parser.parse_json_file(json_file=json_path, allow_extra_keys=True)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        advance_args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]), allow_extra_keys=True)
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
        model_init = getattr(model_module, f"model_init")
    except AttributeError as e:
        raise NotImplementedError(
            f"model_init is not exist. please check it in "
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

    try:
        trainer_module = importlib.import_module(".".join([
            "models",
            f"{model_segment_name}",
            f"trainer_{model_segment_name}",
        ]))
        trainer: Trainer = getattr(trainer_module, "ModelTrainer")
    except AttributeError as e:
        raise NotImplementedError(
            f"ModelTrainer is not exist. please check it in "
            f"{'/'.join(['./models', model_segment_name, 'trainer_'+model_segment_name])}"
        ) from e

    parser = HfArgumentParser((argument, TrainingArguments))
    if json_path:
        model_args, training_args = parser.parse_json_file(json_file=json_path, allow_extra_keys=True)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]), allow_extra_keys=True)
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    try:
        data_collator = getattr(collection_module, "data_collator")
        data_collator = data_collator(**model_args.collator_param)
    except AttributeError as e:
        data_collator = default_data_collator

    model = model_init(model, **model_args.model_init_param)

    collection = collection(**model_args.collection_param)
    train_dataset, dev_dataset = collection.collect()

    trainer = trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator
    )
    if training_args.do_train:
        trainer.train()
        trainer.save_model()

    if training_args.do_eval:
        trainer.evaluate()



if __name__ == "__main__":
    # main("args/cl_experiment/tnews_futher_bert.json")
    main()