# -*- encoding: utf-8 -*-
'''
@create_time: 2021/05/28 11:07:32
@author: lichunyu
'''

import dataclasses
from dataclasses import dataclass
import logging
import os
import sys
from typing import Optional, Dict, Union, Any

from datasets import ClassLabel, load_dataset, load_metric, Dataset
from sklearn.model_selection import train_test_split
from transformers.file_utils import PaddingStrategy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    HfArgumentParser,
    Trainer,
    set_seed,
    PreTrainedTokenizerFast,
    PreTrainedTokenizerBase
)

from transformers.trainer_utils import get_last_checkpoint, is_main_process, is_torch_cuda_available
from transformers.utils import check_min_version
import transformers



class InvoiceTrainer(Trainer):
    ...


logger = logging.getLogger(__name__)


def normalize_bbox(bbox, width, height):
     return [
         int(1000 * (bbox[0] / width)),
         int(1000 * (bbox[1] / height)),
         int(1000 * (bbox[2] / width)),
         int(1000 * (bbox[3] / height)),
     ]

@dataclass
class DataCollatorForKeyValueExtraction:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        print(features)
        input_ids = []
        attention_masks = []
        labels = []
        bbox = []
        label_name = "label" if "label" in features[0].keys() else "target"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        has_image_input = "image" in features[0]
        has_bbox_input = "bbox" in features[0]

        for feat in features:
            token_boxes = []

            for w, box in zip(feat['text'], feat['coor']):
                word_tokens = self.tokenizer.tokenize(w)
                token_boxes.extend([normalize_bbox(box, feat['width'], feat['height'])] * len(word_tokens))

                if len(token_boxes) < self.max_length-2:
                    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]] + (self.max_length-2-len(token_boxes))*[[0,0,0,0]]
                else:
                    token_boxes = [[0, 0, 0, 0]] + token_boxes[:self.max_length-2] + [[1000, 1000, 1000, 1000]]

                batch = self.tokenizer(
                    ' '.join(feat['text']),
                    add_special_tokens = True,
                    truncation='longest_first',
                    max_length = self.max_length,
                    padding = 'max_length',
                    return_attention_mask = True,
                    return_tensors = 'pt',
                    )

                bbox.append(token_boxes)
                input_ids.append(batch['input_ids'])
                attention_masks.append(batch['attention_mask'])
    
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        bbox = torch.tensor(bbox)

        # return input_ids, attention_masks, labels, bbox
        return {
            'input_ids': input_ids,
            'attention_masks': attention_masks,
            'labels': labels,
            'bbox': bbox
        }

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = dataclasses.field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = dataclasses.field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = dataclasses.field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = dataclasses.field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = dataclasses.field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = dataclasses.field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments():

    task_name: Optional[str] = dataclasses.field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})

    author: str = dataclasses.field(default='earlee')

    specific_num_labels: int = dataclasses.field(default=8, metadata={"help": "specific the num_labels if you want"})  # TODO auto load by dataset

    pickle_pth: str = dataclasses.field(default='/Users/lichunyu/Desktop/EarleeNLP/data/wanli_classify_auged_droped_0219.pkl', \
        metadata={'help': 'specific the path where the pickle file which created from pandas is'})

    pad_to_max_length: int = dataclasses.field(default=510)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((DataTrainingArguments, TrainingArguments, ModelArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, training_args, model_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters:  {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    num_labels = data_args.specific_num_labels


    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            # examples[text_column_name],
            padding=padding,
            truncation=True,
            return_overflowing_tokens=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        bboxes = []
        images = []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            label = examples[label_column_name][org_batch_index]
            bbox = examples["bboxes"][org_batch_index]
            image = examples["image"][org_batch_index]
            previous_word_idx = None
            label_ids = []
            bbox_inputs = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    bbox_inputs.append(bbox[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                    bbox_inputs.append(bbox[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
            bboxes.append(bbox_inputs)
            images.append(image)
        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["image"] = images
        return tokenized_inputs

    # metric = load_metric("seqeval")

    def compute_metrics(p):
        return {
                "precision": 1.,
                "recall": 1.,
                "f1": 1.,
                "accuracy": 1.,
            }
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        logger.info('compute_metrics start')

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        # pad_to_multiple_of=8 if training_args.fp16 else None,
        # padding=padding,
        max_length=data_args.pad_to_max_length,
    )

    pickle_pth = data_args.pickle_pth
    df = pd.read_pickle(pickle_pth)
    df_train, df_eval = train_test_split(df, test_size=0.2)
    train_dataset = Dataset.from_pandas(df_train)
    eval_dataset = Dataset.from_pandas(df_eval)
    try:
        print(len(train_dataset))
        print(len(df_train))
        logger.info('=================={}==============='.format(train_dataset[0]))
    except:
        pass

    # Initialize our Trainer
    trainer = InvoiceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


    # if training_args.do_train:
    #     if "train" not in datasets:
    #         raise ValueError("--do_train requires a train dataset")
    #     train_dataset = datasets["train"]
    #     if data_args.max_train_samples is not None:
    #         train_dataset = train_dataset.select(range(data_args.max_train_samples))
    #     train_dataset = train_dataset.map(
    #         tokenize_and_align_labels,
    #         batched=True,
    #         remove_columns=remove_columns,
    #         num_proc=data_args.preprocessing_num_workers,
    #         load_from_cache_file=not data_args.overwrite_cache,
    #     )

    # if training_args.do_eval:
    #     if "validation" not in datasets:
    #         raise ValueError("--do_eval requires a validation dataset")
    #     eval_dataset = datasets["validation"]
    #     if data_args.max_val_samples is not None:
    #         eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
    #     eval_dataset = eval_dataset.map(
    #         tokenize_and_align_labels,
    #         batched=True,
    #         remove_columns=remove_columns,
    #         num_proc=data_args.preprocessing_num_workers,
    #         load_from_cache_file=not data_args.overwrite_cache,
    #     )

    # if training_args.do_predict:
    #     if "test" not in datasets:
    #         raise ValueError("--do_predict requires a test dataset")
    #     test_dataset = datasets["test"]
    #     if data_args.max_test_samples is not None:
    #         test_dataset = test_dataset.select(range(data_args.max_test_samples))
    #     test_dataset = test_dataset.map(
    #         tokenize_and_align_labels,
    #         batched=True,
    #         remove_columns=remove_columns,
    #         num_proc=data_args.preprocessing_num_workers,
    #         load_from_cache_file=not data_args.overwrite_cache,
    #     )


    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    # if training_args.do_predict:
    #     logger.info("*** Predict ***")

        # predictions, labels, metrics = trainer.predict(test_dataset)
        # predictions = np.argmax(predictions, axis=2)

        # # Remove ignored index (special tokens)
        # true_predictions = [
        #     [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        #     for prediction, label in zip(predictions, labels)
        # ]

        # trainer.log_metrics("test", metrics)
        # trainer.save_metrics("test", metrics)

        # # Save predictions
        # output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        # if trainer.is_world_process_zero():
        #     with open(output_test_predictions_file, "w") as writer:
        #         for prediction in true_predictions:
        #             writer.write(" ".join(prediction) + "\n")



if __name__ == '__main__':
    main()