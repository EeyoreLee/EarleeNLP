# -*- encoding: utf-8 -*-
'''
@create_time: 2022/06/24 10:55:36
@author: lichunyu
'''
import json
import sys
sys.path.append(".")

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from transformers import (
    AutoTokenizer,
    BertTokenizer
)
from datasets import load_dataset

from models._base.base_collection import (
    BaseCollection,
    ClassificationDataset,
    BaseTokenization
)
from models._base.base_collator import (
    BaseDataCollator
)

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Collection(BaseCollection):

    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)
        self._init_tokenzier()
        self.vocab = Vocabulary()

    def _init_tokenzier(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)

    def collect(self):
        data_files = {
            "train": self.data_path if self.uncut else self.train_data_path
        }
        if self.dev_data_path is not None:
            data_files["dev"] = self.dev_data_path
        sample = data_files["train"]
        if isinstance(sample, list):
            sample = sample[0]
        extension = sample.split('.')[-1]
        if extension in ["txt", "train", "test", "dev"]:
            extension = "text"
        if extension == "tsv":
            dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
        else:
            dataset = load_dataset(extension, data_files=data_files)

        def prepare(examples):
            text = examples["text"]

        ...

        return dataset


def process_bert(data, tokenizer, vocab):

    bert_inputs = []
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    sent_length = []

    for index, instance in enumerate(data):
        if len(instance['sentence']) == 0:
            continue

        tokens = [tokenizer.tokenize(word) for word in instance['sentence']]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])

        length = len(instance['sentence'])
        _grid_labels = np.zeros((length, length), dtype=np.int)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
        _dist_inputs = np.zeros((length, length), dtype=np.int)
        _grid_mask2d = np.ones((length, length), dtype=np.bool)

        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k

        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19

        for entity in instance["ner"]:
            index = entity["index"]
            for i in range(len(index)):
                if i + 1 >= len(index):
                    break
                _grid_labels[index[i], index[i + 1]] = 1
            _grid_labels[index[-1], index[0]] = vocab.label_to_id(entity["type"])

        _entity_text = set([convert_index_to_text(e["index"], vocab.label_to_id(e["type"]))
                            for e in instance["ner"]])

        sent_length.append(length)
        bert_inputs.append(_bert_inputs)
        grid_labels.append(_grid_labels)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)
        entity_text.append(_entity_text)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def data_collator(data):
    bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text = map(list, zip(*data))

    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)
    labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    # return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text
    return dict(
        bert_inputs=bert_inputs,
        grid_labels=grid_labels,
        grid_mask2d=grid_mask2d,
        pieces2word=pieces2word,
        dist_inputs=dist_inputs,
        sent_length=sent_length,
        entity_text=entity_text
    )


if __name__ == "__main__":
    collection = Collection(
        train_data_path="/disk/223/person/lichunyu/datasets/public_data/weibo_ner/weiboNER_2nd_conll.train.json",
        dev_data_path="/disk/223/person/lichunyu/datasets/public_data/weibo_ner/weiboNER_2nd_conll.dev.json",
        tokenizer_name_or_path="/disk/223/person/lichunyu/pretrain-models/bert-base-chinese"
    )
    datasets = collection.collect()
    ...