# -*- encoding: utf-8 -*-
'''
@create_time: 2021/08/06 17:52:52
@author: lichunyu
'''
import torch
from collections import defaultdict


def ner_entity_f1(pred, target, seq_len, other_label=2, special_idx=[0,1], label2idx=None):
    """按抽取结果计算 F1值， 非 span F值

    :param pred: [description]
    :type pred: [type]
    :param target: [description]
    :type target: [type]
    :param seq_len: [description]
    :type seq_len: [type]
    """

    batch_size = len(seq_len)
    pass


def ner_span_f1(pred, target, seq_len, other_label=2, specical_idx=[0,1], label2idx=None):
    pass



def ner_extract(pred, seq_len, text, other_label=2, special_idx=None, label2idx:dict=None, idx2label:dict=None):
    """暂时只支持BIO

    :param pred: [description]
    :type pred: [type]
    :param seq_len: [description]
    :type seq_len: [type]
    :param other_label: [description], defaults to 2
    :type other_label: int, optional
    :param special_idx: [description], defaults to [0,1]
    :type special_idx: list, optional
    :param label2idx: [description], defaults to None
    :type label2idx: dict, optional
    """
    if special_idx is None:
        special_idx = [0, 1]
    non_label_idx = special_idx + [other_label]
    if idx2label is None:
        idx2label = {v: k for k, v in label2idx.items()}
    batch_size = len(seq_len)
    result = defaultdict(dict)
    for batch_idx in range(batch_size):
        label_stack = []
        location_stack = []
        pred_item = pred[batch_idx][:seq_len[batch_idx]].cpu().tolist()
        for num_idx, label_idx in enumerate(pred_item):
            if label_idx not in non_label_idx:
                if label_stack and idx2label[label_idx][0] == 'B':
                    if label_stack[0][2:] in result[text[batch_idx]]:
                        result[text[batch_idx]][label_stack[0][2:]].append(text[batch_idx][location_stack[0]: location_stack[-1]+1])
                    else:
                        result[text[batch_idx]][label_stack[0][2:]] = []
                        result[text[batch_idx]][label_stack[0][2:]].append(text[batch_idx][location_stack[0]: location_stack[-1]+1])
                    label_stack = []
                    location_stack = []
                label_stack.append(idx2label[label_idx])
                location_stack.append(num_idx)
        if label_stack:
            if label_stack[0][2:] in result[text[batch_idx]]:
                result[text[batch_idx]][label_stack[0][2:]].append(text[batch_idx][location_stack[0]: location_stack[-1]+1])
            else:
                result[text[batch_idx]][label_stack[0][2:]] = []
                result[text[batch_idx]][label_stack[0][2:]].append(text[batch_idx][location_stack[0]: location_stack[-1]+1])

    return result