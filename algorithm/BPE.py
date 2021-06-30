# -*- encoding: utf-8 -*-
'''
@create_time: 2021/06/25 16:02:00
@author: lichunyu
'''

from typing import Optional
from math import inf


vocab_origin = {
    "lower": 2,
    "lover": 3,
    "newest": 6,
    "widest": 4,
    "low": 4
}

def word2char(vocab: dict):
    vocab_char = {}
    for word, num in vocab.items():
        string_char = ' '.join(word) + '<\w>'
        vocab[string_char] = num
    return vocab_char


def get_subword_num(vocab: dict, is_compute_w: bool=True):
    num = 0
    for string_char, _ in vocab.items():
        row_space_num = string_char.count(' ')
        if is_compute_w is True:
            row_space_num += 1
        num += row_space_num
    return num


def gen_subword_pair_list(subword_list:list):
    subword_pair_list = []
    for idx in range(len(subword_list)-1):
        subword_pair_list.append((subword_list[idx], subword_list[idx+1]))
    return subword_pair_list


def get_frequency_of_subword(vocab: dict, is_sort: bool=False):
    freq = {}
    for string_char, word_freq in vocab.items():
        subword_list = string_char.split(' ')
        subword_pair_list = gen_subword_pair_list(subword_list)
        for s in subword_pair_list:
            if s in freq:
                freq[s] += word_freq
            else:
                freq[s] = word_freq

    if is_sort is True:
        freq = {k: v for (k, v) in sorted(freq.items(), key=lambda e:e[1], reverse=True)}
    return freq


def bpe(vocab, vocab_num: Optional[int]=None):
    subword_num = get_subword_num(vocab)
    if vocab_num is not None and subword_num <= vocab_num:
        return vocab

    if vocab_num is None:
        vocab_num = inf

    while subword_num < vocab_num or subword_num > len(vocab):
        frequency_of_subword = get_frequency_of_subword(vocab)