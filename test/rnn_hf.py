# -*- encoding: utf-8 -*-
'''
@create_time: 2022/03/14 17:58:54
@author: lichunyu
'''
import json
import sys
import os
from dataclasses import dataclass, field

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence, pad_sequence
import torch
from torch.utils.data import DataLoader, Dataset

from transformers import (
    BertModel,
    BertTokenizer,
    BertConfig,
    Trainer,
    HfArgumentParser,
    TrainingArguments
)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# torch.no_deprecation_warning = True



class BertRnnClassification(nn.Module):

    def __init__(self, bert_config_or_path, rnn_hidden_size, num_labels=18, need_pack=False):
        super().__init__()
        self.need_pack = need_pack
        self.bert_config = BertConfig.from_pretrained(bert_config_or_path)
        self.bert = BertModel.from_pretrained(bert_config_or_path)
        self.rnn = nn.RNN(input_size=self.bert_config.hidden_size, hidden_size=rnn_hidden_size, batch_first=True)
        self.cls = nn.Linear(rnn_hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = bert_output[0]
        if self.need_pack is False:
            output = self.rnn(sequence_output)
            output = output[0]
        else:
            output = self.rnn(sequence_output)  # TODO: do a job using packed sequence_output to rnn layer
            output = output[0]
        logits = self.cls(output)
        logits = torch.mean(logits, dim=-2)
        loss = None
        if self.training is True:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {"logits": logits, "loss": loss}



# /ai/223/person/lichunyu/datasets/public_data/clue/tnews_public
# |-- dev.json
# |-- labels.json
# |-- test.json
# |-- train.50k.json
# `-- train.json
class BertRnnDataset(Dataset):
    """fixed bert-rnn Dataset for tnews dataset

    Args:
        Dataset (_type_): Dataset from pytorch 1.6.0
    """    

    def __init__(self, tnews_data, tokenizer, max_length=200) -> None:
        super().__init__()
        self.sentences = [_['sentence'] for _ in tnews_data]
        self.labels = [_['label'] for _ in tnews_data]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        sentence = self.sentences[index]
        labels = torch.tensor(int(self.labels[index])-100)
        inputs = tokenizer(
            text=sentence,
            truncation='longest_first',
            add_special_tokens=True,
            max_length = self.max_length,
            padding = 'max_length',
            return_attention_mask = True,
            return_tensors = 'pt',
            )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        token_type_ids = inputs['token_type_ids'].squeeze(0)
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels
        }
        return data


@dataclass
class CustomizeArguments:

    author: str = field(default='lichunyu', metadata={"help": "the author of the code"})


class ETrainer(Trainer):

    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size)

    def get_eval_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=self.args.eval_batch_size)


if __name__ == "__main__":
    parser = HfArgumentParser((CustomizeArguments, TrainingArguments))
    # if json_path:
    #     custom_args, training_args = parser.parse_json_file(json_file=json_path)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        custom_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        custom_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = BertTokenizer.from_pretrained('/ai/223/person/lichunyu/pretrain-models/bert-base-chinese')
    model = BertRnnClassification(
        bert_config_or_path='/ai/223/person/lichunyu/pretrain-models/bert-base-chinese',
        rnn_hidden_size=512
    )
    with open('/ai/223/person/lichunyu/datasets/public_data/clue/tnews_public/train.json') as f:
        train_data = f.read().splitlines()
        train_data = [json.loads(_) for _ in train_data]

    with open('/ai/223/person/lichunyu/datasets/public_data/clue/tnews_public/dev.json') as f:
        dev_data = f.read().splitlines()
        dev_data = [json.loads(_) for _ in dev_data]

    train_dataset = BertRnnDataset(train_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=16)

    dev_dataset = BertRnnDataset(dev_data, tokenizer)
    dev_dataloader = DataLoader(dev_dataset, batch_size=16)

    trainer = ETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        # tokenizer=tokenizer,
    )
    trainer.train()

    ...