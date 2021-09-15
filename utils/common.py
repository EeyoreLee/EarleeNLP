# -*- encoding: utf-8 -*-
'''
@create_time: 2021/08/05 14:28:18
@author: lichunyu
'''

from collections import defaultdict
import datetime
import random

from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np
import torch
import torch.nn as nn


def date_now():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")



def print_info(*inp,islog=True,sep=' '):
    from fastNLP import logger
    if islog:
        print(*inp,sep=sep)
    else:
        inp = sep.join(map(str,inp))
        logger.info(inp)


def text_rm_space(text:str):
    offsets, pointer = [], 0
    for idx, char in enumerate(text):
        offsets.append((pointer, idx))
        if char != ' ':
            pointer += 1
    return text.replace(' ', ''), offsets


def flat_f1(preds, labels):

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, pred_flat, average='micro')


def flat_acc(p, l):
    p_f = np.argmax(p, axis=1).flatten()
    l_f = l.flatten()
    return accuracy_score(l_f, p_f)


def format_time(elapsed):

    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def bio_decode(label_ids, text, mask=None, seq_len=None, label2idx=None, idx2label=None, offsets=None):
    batch = label_ids.shape[0]
    if isinstance(text, str):
        text = [text]
    if idx2label is None:
        idx2label = {v: k for k, v in label2idx.items()}
    if seq_len is None:
        seq_len = np.sum(mask, axis=-1) - 2
    if offsets is None:
        offsets = []
        for i in text:
            offsets.append([(idx, idx) for idx, _ in enumerate(i)])

    result = defaultdict(dict)

    for batch_idx in range(batch):
        seq_idx = label_ids[batch_idx][1:1+seq_len[batch_idx]]
        seq_label = [idx2label[_] for _ in seq_idx]
        label_stack = []
        location_stack = []
        for num_idx, label in enumerate(seq_label):
            if label_stack and label[0] == 'O':
                if label_stack[0][2:] in result[text[batch_idx]]:
                    result[text[batch_idx]][label_stack[0][2:]].append(text[batch_idx][location_stack[0]: location_stack[-1]+1])
                else:
                    result[text[batch_idx]][label_stack[0][2:]] = []
                    result[text[batch_idx]][label_stack[0][2:]].append(text[batch_idx][location_stack[0]: location_stack[-1]+1])
                label_stack = []
                location_stack = []
                continue
            if label != 'O':
                # if label_stack and (label[0] == 'B' or label.split('-')[-1] != label_stack[-1].split('-')[-1]):
                if label_stack and label[0] == 'B':
                    if label_stack[0][2:] in result[text[batch_idx]]:
                        result[text[batch_idx]][label_stack[0][2:]].append(text[batch_idx][location_stack[0]: location_stack[-1]+1])
                    else:
                        result[text[batch_idx]][label_stack[0][2:]] = []
                        result[text[batch_idx]][label_stack[0][2:]].append(text[batch_idx][location_stack[0]: location_stack[-1]+1])
                    label_stack = []
                    location_stack = []
                label_stack.append(label)
                location_stack.append(offsets[batch_idx][num_idx][-1])
        if label_stack:
            if label_stack[0][2:] in result[text[batch_idx]]:
                result[text[batch_idx]][label_stack[0][2:]].append(text[batch_idx][location_stack[0]: location_stack[-1]+1])
            else:
                result[text[batch_idx]][label_stack[0][2:]] = []
                result[text[batch_idx]][label_stack[0][2:]].append(text[batch_idx][location_stack[0]: location_stack[-1]+1])

    return result


def seq_idx2label(label_ids, mask=None, seq_len=None, label2idx=None, idx2label=None):
    batch = label_ids.shape[0]
    if idx2label is None:
        idx2label = {v: k for k, v in label2idx.items()}
    if seq_len is None:
        seq_len = np.sum(mask, axis=-1) - 2

    res = []
    for batch_idx in range(batch):
        seq_idx = label_ids[batch_idx][1:1+seq_len[batch_idx]]
        seq_label = [idx2label[_] for _ in seq_idx]
        res.append(seq_label)

    return res


def get_piece_randint(start, end, n):
    res = []
    if (end - start + 1) < n:
        raise Exception('范围内没足够整数')
    while len(res) < n:
        num = random.randint(start, end)
        if num not in res:
            res.append(num)
    return res


def viterbi_ensemble_decode(logits, mask, unpad=False, include_start_end_trans=False, start_scores=None, end_scores=None, trans_m=None, _constrain=None, device='cuda:0', ensemble_method='sum'):
    """给定一个特征矩阵以及转移分数矩阵，计算出最佳的路径以及对应的分数

    :param torch.FloatTensor logits: batch_size x max_len x num_tags，特征矩阵。
    :param torch.ByteTensor mask: batch_size x max_len, 为0的位置认为是pad；如果为None，则认为没有padding。
    :param bool unpad: 是否将结果删去padding。False, 返回的是batch_size x max_len的tensor; True，返回的是
        List[List[int]], 内部的List[int]为每个sequence的label，已经除去pad部分，即每个List[int]的长度是这
        个sample的有效长度。
    :return: 返回 (paths, scores)。
                paths: 是解码后的路径, 其值参照unpad参数.
                scores: torch.FloatTensor, size为(batch_size,), 对应每个最优路径的分数。

    """
    logits = [i.to(device) for i in logits]
    mask = [i.to(device) for i in mask]
    trans_m = [i.to(device) for i in trans_m]
    batch_size, seq_len, n_tags = logits[0].size()
    logits = [i.transpose(0, 1).data for i in logits]  # L, B, H
    # mask = [i.transpose(0, 1).data.eq(1) for i in mask]  # L, B
    mask = mask[0].transpose(0,1).data.eq(1)
    if ensemble_method == 'sum':
        logits_b = logits[0]
        for i in logits[1:]:
            logits_b += i
        logits = logits_b

        trans_m_b = trans_m[0]
        for i in trans_m[1:]:
            trans_m_b += i
        trans_m = trans_m_b

    # dp
    vpath = logits.new_zeros((seq_len, batch_size, n_tags), dtype=torch.long)
    vscore = logits[0]
    if _constrain is None:
        constrain = torch.zeros(n_tags+2, n_tags+2)
        _constrain = nn.Parameter(constrain, requires_grad=False)
    transitions = _constrain.data.clone().to(device)
    transitions[:n_tags, :n_tags] += trans_m.data
    if include_start_end_trans:
        transitions[n_tags, :n_tags] += start_scores.data
        transitions[:n_tags, n_tags + 1] += end_scores.data

    vscore += transitions[n_tags, :n_tags]
    trans_score = transitions[:n_tags, :n_tags].view(1, n_tags, n_tags).data
    for i in range(1, seq_len):
        prev_score = vscore.view(batch_size, n_tags, 1)
        cur_score = logits[i].view(batch_size, 1, n_tags)
        score = prev_score + trans_score + cur_score
        best_score, best_dst = score.max(1)
        vpath[i] = best_dst
        vscore = best_score.masked_fill(mask[i].eq(0).view(batch_size, 1), 0) + \
                    vscore.masked_fill(mask[i].view(batch_size, 1), 0)

    if include_start_end_trans:
        vscore += transitions[:n_tags, n_tags + 1].view(1, -1)

    # backtrace
    batch_idx = torch.arange(batch_size, dtype=torch.long, device=logits.device)
    seq_idx = torch.arange(seq_len, dtype=torch.long, device=logits.device)
    lens = (mask.long().sum(0) - 1)
    # idxes [L, B], batched idx from seq_len-1 to 0
    idxes = (lens.view(1, -1) - seq_idx.view(-1, 1)) % seq_len

    ans = logits.new_empty((seq_len, batch_size), dtype=torch.long)
    ans_score, last_tags = vscore.max(1)
    ans[idxes[0], batch_idx] = last_tags
    for i in range(seq_len - 1):
        last_tags = vpath[idxes[i], batch_idx, last_tags]
        ans[idxes[i + 1], batch_idx] = last_tags
    ans = ans.transpose(0, 1)
    if unpad:
        paths = []
        for idx, seq_len in enumerate(lens):
            paths.append(ans[idx, :seq_len + 1].tolist())
    else:
        paths = ans
    return paths, ans_score




if __name__ == '__main__':
    # print(text_rm_space('as fas ,fsfsf asdf'))
    label_ids = np.array([
        [0,0,0,15,3,3,0,0,0,0,0],
        [0,0,2,1,1,1,0,0,0,0,0],
        [0,8,9,0,0,0,0,0,0,0,0]
    ])
    mask = np.array(
        [
            [1,1,1,1,1,1,1,0,0,0,0],
            [1,1,1,1,1,1,1,1,0,0,0],
            [1,1,1,1,1,1,1,1,0,0,0]
        ]
    )
    label2idx = {'O': 0,
        'I-Video-Play-name': 1,
        'I-Radio-Listen-channel': 2,
        'I-Calendar-Query-datetime_date': 3,
        'I-Alarm-Update-notes': 4,
        'I-FilmTele-Play-name': 5,
        'I-Alarm-Update-datetime_time': 6,
        'I-Radio-Listen-name': 7,
        'I-Alarm-Update-datetime_date': 8,
        'I-HomeAppliance-Control-appliance': 9,
        'I-Travel-Query-destination': 10,
        'I-HomeAppliance-Control-details': 11,
        'I-Radio-Listen-frequency': 12,
        'I-Music-Play-song': 13,
        'I-Weather-Query-datetime_date': 14,
        'B-Calendar-Query-datetime_date': 15,
        'I-Weather-Query-city': 16,
        'B-HomeAppliance-Control-appliance': 17,
        'B-Travel-Query-destination': 18,
        'I-Video-Play-datetime_date': 19,
        'I-Music-Play-artist': 20,
        'B-Alarm-Update-datetime_date': 21,
        'B-Weather-Query-city': 22,
        'B-Video-Play-name': 23,
        'B-Weather-Query-datetime_date': 24,
        'B-Alarm-Update-datetime_time': 25,
        'I-FilmTele-Play-artist': 26,
        'B-Alarm-Update-notes': 27,
        'B-HomeAppliance-Control-details': 28,
        'B-FilmTele-Play-name': 29,
        'B-Radio-Listen-channel': 30,
        'I-Music-Play-age': 31,
        'I-FilmTele-Play-age': 32,
        'B-Radio-Listen-name': 33,
        'I-FilmTele-Play-tag': 34,
        'I-Music-Play-album': 35,
        'B-Music-Play-artist': 36,
        'B-FilmTele-Play-artist': 37,
        'I-Travel-Query-departure': 38,
        'B-Music-Play-song': 39,
        'I-FilmTele-Play-play_setting': 40,
        'I-Travel-Query-datetime_date': 41,
        'B-Travel-Query-departure': 42,
        'I-Radio-Listen-artist': 43,
        'B-FilmTele-Play-tag': 44,
        'I-Travel-Query-datetime_time': 45,
        'B-Radio-Listen-frequency': 46,
        'B-Radio-Listen-artist': 47,
        'I-Video-Play-datetime_time': 48,
        'B-Video-Play-datetime_date': 49,
        'B-Travel-Query-datetime_date': 50,
        'I-FilmTele-Play-region': 51,
        'B-FilmTele-Play-region': 52,
        'B-FilmTele-Play-play_setting': 53,
        'I-TVProgram-Play-name': 54,
        'B-FilmTele-Play-age': 55,
        'B-Travel-Query-datetime_time': 56,
        'B-Music-Play-age': 57,
        'B-Music-Play-album': 58,
        'I-Video-Play-region': 59,
        'B-Video-Play-region': 60,
        'I-Music-Play-instrument': 61,
        'I-Weather-Query-datetime_time': 62,
        'I-TVProgram-Play-channel': 63,
        'B-Music-Play-instrument': 64,
        'I-Audio-Play-name': 65,
        'B-Video-Play-datetime_time': 66,
        'B-Weather-Query-datetime_time': 67,
        'B-TVProgram-Play-name': 68,
        'I-TVProgram-Play-datetime_date': 69,
        'I-Audio-Play-artist': 70,
        'B-Audio-Play-name': 71,
        'I-TVProgram-Play-datetime_time': 72,
        'B-TVProgram-Play-channel': 73,
        'B-TVProgram-Play-datetime_date': 74,
        'B-Audio-Play-artist': 75,
        'B-TVProgram-Play-datetime_time': 76,
        'I-Audio-Play-play_setting': 77,
        'B-Audio-Play-play_setting': 78,
        'B-Audio-Play-tag': 79,
        'I-Audio-Play-tag': 80}
    bio_decode(label_ids, text=['今天星期五', '今天是星期五', '手机壳就撒开'], mask=mask, label2idx=label2idx)