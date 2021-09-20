# -*- encoding: utf-8 -*-
'''
@create_time: 2021/07/30 14:01:36
@author: lichunyu
'''

from fastNLP import cache_results, Vocabulary
from fastNLP.embeddings.embedding import TokenEmbedding
from fastNLP.io.file_utils import PRETRAIN_STATIC_FILES, _get_embedding_url, cached_path
import os
import warnings
from collections import defaultdict
import collections
from copy import deepcopy
from fastNLP.modules import ConditionalRandomField
import time
import datetime
import pytz
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from functools import partial
from fastNLP import DataSet

from fastNLP.core import logger
from fastNLP.core import Trainer
from fastNLP.modules.utils import _get_file_name_base_on_postfix
from fastNLP.embeddings.contextual_embedding import ContextualEmbedding
from fastNLP.embeddings.bert_embedding import _WordBertModel
from fastNLP.io.file_utils import PRETRAINED_BERT_MODEL_DIR
from fastNLP import LossInForward
from fastNLP.core.metrics import SpanFPreRecMetric,AccuracyMetric
from fastNLP.core.callback import WarmupCallback,GradientClipCallback,EarlyStopCallback
from fastNLP import LRScheduler
from fastNLP import FitlogCallback
from fastNLP.core import Callback

from transformers import TrainingArguments, HfArgumentParser
from transformers.trainer_utils import is_main_process, get_last_checkpoint
from dataclasses import dataclass, field
from fastNLP.io.loader import ConllLoader

from utils.common import print_info

load_dataset_seed = 42


yangjie_rich_pretrain_unigram_path = '/remote-home/xnli/data/pretrain/chinese/gigaword_chn.all.a2b.uni.ite50.vec'
yangjie_rich_pretrain_bigram_path = '/remote-home/xnli/data/pretrain/chinese/gigaword_chn.all.a2b.bi.ite50.vec'
yangjie_rich_pretrain_word_path = '/remote-home/xnli/data/pretrain/chinese/ctb.50d.vec'
yangjie_rich_pretrain_char_and_word_path = '/remote-home/xnli/data/pretrain/chinese/yangjie_word_char_mix.txt'
# lk_word_path = '/remote-home/xnli/data/pretrain/chinese/sgns.merge.word'
lk_word_path_2 = '/remote-home/xnli/data/pretrain/chinese/sgns.merge.word_2'

ontonote4ner_cn_path = '/remote-home/xnli/data/corpus/sequence_labelling/chinese_ner/OntoNote4NER'
msra_ner_cn_path = '/remote-home/xnli/data/corpus/sequence_labelling/chinese_ner/MSRANER'
resume_ner_path = '/remote-home/xnli/data/corpus/sequence_labelling/chinese_ner/ResumeNER'
weibo_ner_path = '/remote-home/xnli/data/corpus/sequence_labelling/chinese_ner/WeiboNER'


def get_bigrams(words):
    result = []
    for i,w in enumerate(words):
        if i!=len(words)-1:
            result.append(words[i]+words[i+1])
        else:
            result.append(words[i]+'<end>')

    return result


class MyDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        assert 0<=p<=1
        self.p = p

    def forward(self, x):
        if self.training and self.p>0.001:
            # print('mydropout!')
            mask = torch.rand(x.size())
            # print(mask.device)
            mask = mask.to(x)
            # print(mask.device)
            mask = mask.lt(self.p)
            x = x.masked_fill(mask, 0)/(1-self.p)
        return x


def size2MB(size_,type_size=4):
    num = 1
    for s in size_:
        num*=s

    return num * type_size /1000 /1000


def get_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
    """Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    rel pos init:
    如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
    如果是1，那么就按-max_len,max_len来初始化
    """
    num_embeddings = 2*max_seq_len+1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    if rel_pos_init == 0:
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len,max_seq_len+1, dtype=torch.float).unsqueeze(1)*emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb



def seq_len_to_mask(seq_len, max_len=None):
    """

    将一个表示sequence length的一维数组转换为二维的mask，不包含的位置为0。
    转变 1-d seq_len到2-d mask.

    .. code-block::
    
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.size())
        torch.Size([14, 15])
        >>> seq_len = np.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.shape)
        (14, 15)
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len, max_len=100)
        >>>print(mask.size())
        torch.Size([14, 100])

    :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
    :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
        区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask


class MultiHead_Attention_rel(nn.Module):
    def __init__(self, hidden_size, num_heads, pe, scaled=True, max_seq_len=-1,
                 dvc=None,mode=collections.defaultdict(bool),k_proj=True,q_proj=True,v_proj=True,r_proj=True,
                 attn_dropout=None,
                 ff_final=True):
        '''

        :param hidden_size:
        :param num_heads:
        :param scaled:
        :param debug:
        :param max_seq_len:
        :param device:
        '''
        super().__init__()
        self.mode=mode
        if self.mode['debug']:
            print_info('rel pos attn')
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        self.max_seq_len = max_seq_len
        if dvc is None:
            dvc = torch.device('cpu')
        self.dvc = dvc
        assert (self.per_head_size * self.num_heads == self.hidden_size)

        self.k_proj=k_proj
        self.q_proj=q_proj
        self.v_proj=v_proj
        self.r_proj=r_proj


        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_r = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_final = nn.Linear(self.hidden_size, self.hidden_size)
        self.u = nn.Parameter(torch.Tensor(self.num_heads,self.per_head_size))
        self.v = nn.Parameter(torch.Tensor(self.num_heads,self.per_head_size))

        self.pe = pe

        self.dropout = MyDropout(attn_dropout)

        if ff_final:
            self.ff_final = nn.Linear(self.hidden_size,self.hidden_size)



    def forward(self,key, query, value, seq_len):
        # B prepare relative position encoding
        max_seq_len = torch.max(seq_len)
        rel_distance = self.seq_len_to_rel_distance(max_seq_len)
        rel_distance_flat = rel_distance.view(-1)
        rel_pos_embedding_flat = self.pe[rel_distance_flat+self.max_seq_len]
        rel_pos_embedding = rel_pos_embedding_flat.view(size=[max_seq_len,max_seq_len,self.hidden_size])
        # E prepare relative position encoding

        if self.k_proj:
            if self.mode['debug']:
                print_info('k_proj!')
            key = self.w_k(key)
        if self.q_proj:
            if self.mode['debug']:
                print_info('q_proj!')
            query = self.w_q(query)
        if self.v_proj:
            if self.mode['debug']:
                print_info('v_proj!')
            value = self.w_v(value)
        if self.r_proj:
            if self.mode['debug']:
                print_info('r_proj!')
            rel_pos_embedding = self.w_r(rel_pos_embedding)

        batch = key.size(0)
        max_seq_len = key.size(1)


        # batch * seq_len * n_head * d_head
        key = torch.reshape(key, [batch, max_seq_len, self.num_heads, self.per_head_size])
        query = torch.reshape(query, [batch, max_seq_len, self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, max_seq_len, self.num_heads, self.per_head_size])
        rel_pos_embedding = torch.reshape(rel_pos_embedding,
                                          [max_seq_len, max_seq_len, self.num_heads,self.per_head_size])


        # batch * n_head * seq_len * d_head
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)



        # batch * n_head * d_head * key_len
        key = key.transpose(-1, -2)


        #A
        A_ = torch.matmul(query,key)

        #B
        rel_pos_embedding_for_b = rel_pos_embedding.unsqueeze(0).permute(0, 3, 1, 4, 2)
        # after above, rel_pos_embedding: batch * num_head * query_len * per_head_size * key_len
        query_for_b = query.view([batch, self.num_heads, max_seq_len, 1, self.per_head_size])
        # print('query for b:{}'.format(query_for_b.size()))
        # print('rel_pos_embedding_for_b{}'.format(rel_pos_embedding_for_b.size()))
        B_ = torch.matmul(query_for_b,rel_pos_embedding_for_b).squeeze(-2)

        #D
        rel_pos_embedding_for_d = rel_pos_embedding.unsqueeze(-2)
        # after above, rel_pos_embedding: query_seq_len * key_seq_len * num_heads * 1 *per_head_size
        v_for_d = self.v.unsqueeze(-1)
        # v_for_d: num_heads * per_head_size * 1
        D_ = torch.matmul(rel_pos_embedding_for_d,v_for_d).squeeze(-1).squeeze(-1).permute(2,0,1).unsqueeze(0)

        #C
        # key: batch * n_head * d_head * key_len
        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)
        # u_for_c: 1(batch broadcast) * num_heads * 1 *per_head_size
        key_for_c = key
        C_ = torch.matmul(u_for_c, key)

        #att_score: Batch * num_heads * query_len * key_len
        # A, B C and D is exactly the shape
        if self.mode['debug']:
            print_info('A:{}'.format(A_.size()))
            print_info('B:{}'.format(B_.size()))
            print_info('C:{}'.format(C_.size()))
            print_info('D:{}'.format(D_.size()))
        attn_score_raw = A_ + B_ + C_ + D_

        if self.scaled:
            attn_score_raw  = attn_score_raw / math.sqrt(self.per_head_size)

        mask = seq_len_to_mask(seq_len).bool().unsqueeze(1).unsqueeze(1)
        attn_score_raw_masked = attn_score_raw.masked_fill(~mask, -1e15)
        if self.mode['debug']:
            print('attn_score_raw_masked:{}'.format(attn_score_raw_masked))
            print('seq_len:{}'.format(seq_len))

        attn_score = F.softmax(attn_score_raw_masked,dim=-1)

        attn_score = self.dropout(attn_score)

        value_weighted_sum = torch.matmul(attn_score, value)

        result = value_weighted_sum.transpose(1,2).contiguous(). \
            reshape(batch, max_seq_len, self.hidden_size)


        if hasattr(self,'ff_final'):
            print('ff_final!!')
            result = self.ff_final(result)

        return result

    def seq_len_to_rel_distance(self,max_seq_len):
        '''

        :param seq_len: seq_len batch
        :return: L*L rel_distance
        '''
        index = torch.arange(0, max_seq_len)
        assert index.size(0) == max_seq_len
        assert index.dim() == 1
        index = index.repeat(max_seq_len, 1)
        offset = torch.arange(0, max_seq_len).unsqueeze(1)
        offset = offset.repeat(1, max_seq_len)
        index = index - offset
        index = index.to(self.dvc)
        return index


class Absolute_SE_Position_Embedding(nn.Module):
    def __init__(self,fusion_func,hidden_size,learnable,mode=collections.defaultdict(bool),pos_norm=False,max_len=5000,):
        '''

        :param fusion_func:暂时只有add和concat(直接拼接然后接线性变换)，
        后续得考虑直接拼接再接非线性变换，和将S和E两个位置做非线性变换再加或拼接
        :param hidden_size:
        :param learnable:
        :param debug:
        :param pos_norm:
        :param max_len:
        '''
        super().__init__()
        self.fusion_func = fusion_func
        assert self.fusion_func in {'add','concat','nonlinear_concat','nonlinear_add','add_nonlinear','concat_nonlinear'}
        self.pos_norm = pos_norm
        self.mode = mode
        self.hidden_size = hidden_size
        pe = get_embedding(max_len,hidden_size)
        pe_sum = pe.sum(dim=-1,keepdim=True)
        if self.pos_norm:
            with torch.no_grad():
                pe = pe / pe_sum
        # pe = pe.unsqueeze(0)
        pe_s = copy.deepcopy(pe)
        pe_e = copy.deepcopy(pe)
        self.pe_s = nn.Parameter(pe_s, requires_grad=learnable)
        self.pe_e = nn.Parameter(pe_e, requires_grad=learnable)
        if self.fusion_func == 'concat':
            self.proj = nn.Linear(self.hidden_size * 3,self.hidden_size)

        if self.fusion_func == 'nonlinear_concat':
            self.pos_proj = nn.Sequential(nn.Linear(self.hidden_size * 2,self.hidden_size),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.hidden_size,self.hidden_size))
            self.proj = nn.Linear(self.hidden_size * 2,self.hidden_size)

        if self.fusion_func == 'nonlinear_add':
            self.pos_proj = nn.Sequential(nn.Linear(self.hidden_size * 2,self.hidden_size),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.hidden_size,self.hidden_size))

        if self.fusion_func == 'concat_nonlinear':
            self.proj = nn.Sequential(nn.Linear(self.hidden_size * 3,self.hidden_size),
                                      nn.LeakyReLU(),
                                      nn.Linear(self.hidden_size,self.hidden_size))

        if self.fusion_func == 'add_nonlinear':
            self.proj = nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size),
                                      nn.LeakyReLU(),
                                      nn.Linear(self.hidden_size,self.hidden_size))


    def forward(self,inp,pos_s,pos_e):
        batch = inp.size(0)
        max_len = inp.size(1)
        pe_s = self.pe_s.index_select(0, pos_s.view(-1)).view(batch,max_len,-1)
        pe_e = self.pe_e.index_select(0, pos_e.view(-1)).view(batch,max_len,-1)

        if self.fusion_func == 'concat':
            inp = torch.cat([inp,pe_s,pe_e],dim=-1)
            output = self.proj(inp)
        elif self.fusion_func == 'add':
            output = pe_s + pe_e + inp
        elif self.fusion_func == 'nonlinear_concat':
            pos = self.pos_proj(torch.cat([pe_s,pe_e],dim=-1))
            output = self.proj(torch.cat([inp,pos],dim=-1))
        elif self.fusion_func == 'nonlinear_add':
            pos = self.pos_proj(torch.cat([pe_s,pe_e],dim=-1))
            output = pos + inp
        elif self.fusion_func == 'add_nonlinear':
            inp = inp + pe_s + pe_e
            output = self.proj(inp)

        elif self.fusion_func == 'concat_nonlinear':
            output = self.proj(torch.cat([inp,pe_s,pe_e],dim=-1))



        return output

        # if self.fusion_func == 'add':
        #     result =

    def get_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        rel pos init:
        如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
        如果是1，那么就按-max_len,max_len来初始化
        """
        num_embeddings = 2 * max_seq_len + 1
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        if rel_pos_init == 0:
            emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        else:
            emb = torch.arange(-max_seq_len, max_seq_len + 1, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb


def get_crf_zero_init(label_size, include_start_end_trans=False, allowed_transitions=None,
                 initial_method=None):

    crf = ConditionalRandomField(label_size, include_start_end_trans)

    crf.trans_m = nn.Parameter(torch.zeros(size=[label_size, label_size], requires_grad=True))
    if crf.include_start_end_trans:
        crf.start_scores = nn.Parameter(torch.zeros(size=[label_size], requires_grad=True))
        crf.end_scores = nn.Parameter(torch.zeros(size=[label_size], requires_grad=True))
    return crf


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_w = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self,w):

        current = self.root
        for c in w:
            current = current.children[c]

        current.is_w = True

    def search(self,w):
        '''

        :param w:
        :return:
        -1:not w route
        0:subroute but not word
        1:subroute and word
        '''
        current = self.root

        for c in w:
            current = current.children.get(c)

            if current is None:
                return -1

        if current.is_w:
            return 1
        else:
            return 0

    def get_lexicon(self,sentence):
        result = []
        for i in range(len(sentence)):
            current = self.root
            for j in range(i, len(sentence)):
                current = current.children.get(sentence[j])
                if current is None:
                    break

                if current.is_w:
                    result.append([i,j,sentence[i:j+1]])

        return result



class Four_Pos_Fusion_Embedding(nn.Module):
    def __init__(self,pe,four_pos_fusion,pe_ss,pe_se,pe_es,pe_ee,max_seq_len,hidden_size,mode):
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.max_seq_len=max_seq_len
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        self.pe = pe
        self.four_pos_fusion = four_pos_fusion
        if self.four_pos_fusion == 'ff':
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size),
                                                    nn.ReLU(inplace=True))
        if self.four_pos_fusion == 'ff_linear':
            self.pos_fusion_forward = nn.Linear(self.hidden_size*4,self.hidden_size)

        elif self.four_pos_fusion == 'ff_two':
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size*2,self.hidden_size),
                                                    nn.ReLU(inplace=True))
        elif self.four_pos_fusion == 'attn':
            self.w_r = nn.Linear(self.hidden_size,self.hidden_size)
            self.pos_attn_score = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size*4),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size*4,4),
                                                nn.Softmax(dim=-1))

            # print('暂时不支持以attn融合pos信息')
        elif self.four_pos_fusion == 'gate':
            self.w_r = nn.Linear(self.hidden_size,self.hidden_size)
            self.pos_gate_score = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size*2),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size*2,4*self.hidden_size))

            # print('暂时不支持以gate融合pos信息')
            # exit(1208)
    def forward(self,pos_s,pos_e):
        batch = pos_s.size(0)
        #这里的seq_len已经是之前的seq_len+lex_num了
        pos_ss = pos_s.unsqueeze(-1)-pos_s.unsqueeze(-2)
        pos_se = pos_s.unsqueeze(-1)-pos_e.unsqueeze(-2)
        pos_es = pos_e.unsqueeze(-1)-pos_s.unsqueeze(-2)
        pos_ee = pos_e.unsqueeze(-1)-pos_e.unsqueeze(-2)

        if self.mode['debug']:
            print('pos_s:{}'.format(pos_s))
            print('pos_e:{}'.format(pos_e))
            print('pos_ss:{}'.format(pos_ss))
            print('pos_se:{}'.format(pos_se))
            print('pos_es:{}'.format(pos_es))
            print('pos_ee:{}'.format(pos_ee))
        # B prepare relative position encoding
        max_seq_len = pos_s.size(1)
        # rel_distance = self.seq_len_to_rel_distance(max_seq_len)

        # rel_distance_flat = rel_distance.view(-1)
        # rel_pos_embedding_flat = self.pe[rel_distance_flat+self.max_seq_len]
        # rel_pos_embedding = rel_pos_embedding_flat.view(size=[max_seq_len,max_seq_len,self.hidden_size])
        pe_ss = self.pe_ss[(pos_ss).view(-1)+self.max_seq_len].view(size=[batch,max_seq_len,max_seq_len,-1])
        pe_se = self.pe_se[(pos_se).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_es = self.pe_es[(pos_es).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])
        pe_ee = self.pe_ee[(pos_ee).view(-1) + self.max_seq_len].view(size=[batch, max_seq_len, max_seq_len, -1])

        # print('pe_ss:{}'.format(pe_ss.size()))

        if self.four_pos_fusion == 'ff':
            pe_4 = torch.cat([pe_ss,pe_se,pe_es,pe_ee],dim=-1)
            if self.mode['gpumm']:
                print('四个位置合起来:{},{}'.format(pe_4.size(),size2MB(pe_4.size())))
            rel_pos_embedding = self.pos_fusion_forward(pe_4)
        if self.four_pos_fusion == 'ff_linear':
            pe_4 = torch.cat([pe_ss,pe_se,pe_es,pe_ee],dim=-1)
            if self.mode['gpumm']:
                print('四个位置合起来:{},{}'.format(pe_4.size(),size2MB(pe_4.size())))
            rel_pos_embedding = self.pos_fusion_forward(pe_4)
        if self.four_pos_fusion == 'ff_two':
            pe_2 = torch.cat([pe_ss,pe_ee],dim=-1)
            if self.mode['gpumm']:
                print('2个位置合起来:{},{}'.format(pe_2.size(),size2MB(pe_2.size())))
            rel_pos_embedding = self.pos_fusion_forward(pe_2)
        elif self.four_pos_fusion == 'attn':
            pe_4 = torch.cat([pe_ss,pe_se,pe_es,pe_ee],dim=-1)
            attn_score = self.pos_attn_score(pe_4)
            pe_4_unflat = self.w_r(pe_4.view(batch,max_seq_len,max_seq_len,4,self.hidden_size))
            pe_4_fusion = (attn_score.unsqueeze(-1) * pe_4_unflat).sum(dim=-2)
            rel_pos_embedding = pe_4_fusion
            if self.mode['debug']:
                print('pe_4照理说应该是 Batch * SeqLen * SeqLen * HiddenSize')
                print(pe_4_fusion.size())

        elif self.four_pos_fusion == 'gate':
            pe_4 = torch.cat([pe_ss, pe_se, pe_es, pe_ee], dim=-1)
            gate_score = self.pos_gate_score(pe_4).view(batch,max_seq_len,max_seq_len,4,self.hidden_size)
            gate_score = F.softmax(gate_score,dim=-2)
            pe_4_unflat = self.w_r(pe_4.view(batch, max_seq_len, max_seq_len, 4, self.hidden_size))
            pe_4_fusion = (gate_score * pe_4_unflat).sum(dim=-2)
            rel_pos_embedding = pe_4_fusion


        return rel_pos_embedding



class MultiHead_Attention_Lattice_rel_save_gpumm(nn.Module):
    def __init__(self, hidden_size, num_heads, pe,
                 pe_ss,pe_se,pe_es,pe_ee,
                 scaled=True, max_seq_len=-1,
                 dvc=None,mode=collections.defaultdict(bool),k_proj=True,q_proj=True,v_proj=True,r_proj=True,
                 attn_dropout=None,
                 ff_final=True,
                 four_pos_fusion=None):
        '''

        :param hidden_size:
        :param num_heads:
        :param scaled:
        :param debug:
        :param max_seq_len:
        :param device:
        '''
        super().__init__()
        assert four_pos_fusion is not None
        self.four_pos_fusion = four_pos_fusion
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        self.mode = mode
        if self.mode['debug']:
            print_info('rel pos attn')
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        self.max_seq_len = max_seq_len
        if dvc is None:
            dvc = torch.device('cpu')
        self.dvc = dvc
        assert (self.per_head_size * self.num_heads == self.hidden_size)

        self.k_proj=k_proj
        self.q_proj=q_proj
        self.v_proj=v_proj
        self.r_proj=r_proj

        if self.four_pos_fusion == 'ff':
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size),
                                                    nn.ReLU(inplace=True))
        elif self.four_pos_fusion == 'attn':
            self.pos_attn_score = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size*4),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size*4,4),
                                                nn.Softmax(dim=-1))

            # print('暂时不支持以attn融合pos信息')
        elif self.four_pos_fusion == 'gate':
            self.pos_gate_score = nn.Sequential(nn.Linear(self.hidden_size*4,self.hidden_size*2),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_size*2,4*self.hidden_size))

            # print('暂时不支持以gate融合pos信息')
            # exit(1208)


        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_r = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_final = nn.Linear(self.hidden_size, self.hidden_size)
        self.u = nn.Parameter(torch.Tensor(self.num_heads,self.per_head_size))
        self.v = nn.Parameter(torch.Tensor(self.num_heads,self.per_head_size))

        self.pe = pe

        self.dropout = MyDropout(attn_dropout)

        if ff_final:
            self.ff_final = nn.Linear(self.hidden_size,self.hidden_size)



    def forward(self,key, query, value, seq_len, lex_num, pos_s,pos_e,rel_pos_embedding):
        batch = key.size(0)

        if self.k_proj:
            if self.mode['debug']:
                print_info('k_proj!')
            key = self.w_k(key)
        if self.q_proj:
            if self.mode['debug']:
                print_info('q_proj!')
            query = self.w_q(query)
        if self.v_proj:
            if self.mode['debug']:
                print_info('v_proj!')
            value = self.w_v(value)
        if self.r_proj:
            if self.mode['debug']:
                print_info('r_proj!')
            rel_pos_embedding = self.w_r(rel_pos_embedding)

        batch = key.size(0)
        max_seq_len = key.size(1)


        # batch * seq_len * n_head * d_head
        key = torch.reshape(key, [batch, max_seq_len, self.num_heads, self.per_head_size])
        query = torch.reshape(query, [batch, max_seq_len, self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, max_seq_len, self.num_heads, self.per_head_size])
        rel_pos_embedding = torch.reshape(rel_pos_embedding,
                                          [batch,max_seq_len, max_seq_len, self.num_heads,self.per_head_size])


        # batch * n_head * seq_len * d_head
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)



        # batch * n_head * d_head * key_len
        key = key.transpose(-1, -2)
        # #A
        # A_ = torch.matmul(query,key)
        # #C
        # # key: batch * n_head * d_head * key_len
        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)
        # u_for_c: 1(batch broadcast) * num_heads * 1 *per_head_size
        # key_for_c = key
        # C_ = torch.matmul(u_for_c, key)
        query_and_u_for_c = query + u_for_c
        if self.mode['debug']:
            print('query:{}'.format(query.size()))
            print('u_for_c:{}'.format(u_for_c.size()))
            print('query_and_u_for_c:{}'.format(query_and_u_for_c.size()))
            print('key:{}'.format(key.size()))
        A_C = torch.matmul(query_and_u_for_c, key)

        if self.mode['debug']:
            print('query size:{}'.format(query.size()))
            print('query_and_u_for_c size:{}'.format(query_and_u_for_c.size()))

        #B
        rel_pos_embedding_for_b = rel_pos_embedding.permute(0, 3, 1, 4, 2)
        # after above, rel_pos_embedding: batch * num_head * query_len * per_head_size * key_len
        query_for_b = query.view([batch, self.num_heads, max_seq_len, 1, self.per_head_size])
        # after above, query_for_b: batch * num_head * query_len * 1 * per_head_size
        # print('query for b:{}'.format(query_for_b.size()))
        # print('rel_pos_embedding_for_b{}'.format(rel_pos_embedding_for_b.size()))
        # B_ = torch.matmul(query_for_b,rel_pos_embedding_for_b).squeeze(-2)

        #D
        # rel_pos_embedding_for_d = rel_pos_embedding.unsqueeze(-2)
        # after above, rel_pos_embedding: batch * query_seq_len * key_seq_len * num_heads * 1 *per_head_size
        # v_for_d = self.v.unsqueeze(-1)
        # v_for_d: num_heads * per_head_size * 1
        # D_ = torch.matmul(rel_pos_embedding_for_d,v_for_d).squeeze(-1).squeeze(-1).permute(0,3,1,2)

        query_for_b_and_v_for_d = query_for_b + self.v.view(1,self.num_heads,1,1,self.per_head_size)
        B_D = torch.matmul(query_for_b_and_v_for_d, rel_pos_embedding_for_b).squeeze(-2)
        #att_score: Batch * num_heads * query_len * key_len
        # A, B C and D is exactly the shape
        if self.mode['debug']:
            print_info('AC:{}'.format(A_C.size()))
            print_info('BD:{}'.format(B_D.size()))
            # print_info('A:{}'.format(A_.size()))
            # print_info('B:{}'.format(B_.size()))
            # print_info('C:{}'.format(C_.size()))
            # print_info('D:{}'.format(D_.size()))
        attn_score_raw = A_C + B_D

        if self.scaled:
            attn_score_raw  = attn_score_raw / math.sqrt(self.per_head_size)

        mask = seq_len_to_mask(seq_len+lex_num).bool().unsqueeze(1).unsqueeze(1)
        attn_score_raw_masked = attn_score_raw.masked_fill(~mask, -1e15)
        if self.mode['debug']:
            print('attn_score_raw_masked:{}'.format(attn_score_raw_masked))
            print('seq_len:{}'.format(seq_len))

        attn_score = F.softmax(attn_score_raw_masked,dim=-1)

        attn_score = self.dropout(attn_score)

        value_weighted_sum = torch.matmul(attn_score, value)

        result = value_weighted_sum.transpose(1,2).contiguous(). \
            reshape(batch, max_seq_len, self.hidden_size)


        if hasattr(self,'ff_final'):
            print('ff_final!!')
            result = self.ff_final(result)

        return result

    def seq_len_to_rel_distance(self,max_seq_len):
        '''

        :param seq_len: seq_len batch
        :return: L*L rel_distance
        '''
        index = torch.arange(0, max_seq_len)
        assert index.size(0) == max_seq_len
        assert index.dim() == 1
        index = index.repeat(max_seq_len, 1)
        offset = torch.arange(0, max_seq_len).unsqueeze(1)
        offset = offset.repeat(1, max_seq_len)
        index = index - offset
        index = index.to(self.dvc)
        return index



class MultiHead_Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, scaled=True,mode=collections.defaultdict(bool), k_proj=True,q_proj=True,v_proj=True,
                 attn_dropout=None,ff_final=True):
        super().__init__()
        #这个模型接受的输入本身是带有位置信息的，适用于transformer的绝对位置编码模式
        # TODO: attention dropout
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        assert (self.per_head_size * self.num_heads == self.hidden_size)
        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        if ff_final:
            self.ff_final = nn.Linear(self.hidden_size, self.hidden_size)

        self.mode = mode
        self.k_proj=k_proj
        self.q_proj=q_proj
        self.v_proj=v_proj
        if self.mode['debug']:
            print_info('abs pos attn')

        if attn_dropout == None:
            dropout = collections.defaultdict(int)
        self.dropout = MyDropout(attn_dropout)


    def forward(self, key, query, value, seq_len, lex_num=0):
        if self.k_proj:
            key = self.w_k(key)
        if self.q_proj:
            query = self.w_q(query)
        if self.v_proj:
            value = self.w_v(value)

        batch = key.size(0)
        max_seq_len = key.size(1)

        key = torch.reshape(key, [batch, max_seq_len, self.num_heads, self.per_head_size])
        query = torch.reshape(query, [batch, max_seq_len, self.num_heads, self.per_head_size])
        value = torch.reshape(value, [batch, max_seq_len, self.num_heads, self.per_head_size])

        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        key = key.transpose(-1, -2)

        attention_raw = torch.matmul(query, key)

        if self.scaled:
            attention_raw = attention_raw / math.sqrt(self.per_head_size)

        # if self.mode['debug']:
        #     print('attention_raw:{}'.format(attention_raw.size()))
        #     print('mask:{},{}'.format(mask.size(),mask.dtype))
        #     print('mask==0:{}'.format((mask==0).dtype))
        mask = seq_len_to_mask(seq_len + lex_num).bool().unsqueeze(1).unsqueeze(1)
        attention_raw_masked = attention_raw.masked_fill(~mask, -1e15)

        attn_score = F.softmax(attention_raw_masked, dim=-1)
        attn_score = self.dropout(attn_score)
        # TODO attention dropout

        value_weighted_sum = torch.matmul(attn_score, value)

        result = value_weighted_sum.transpose(1, 2).contiguous(). \
            reshape(batch, max_seq_len, self.hidden_size)

        if hasattr(self,'ff_final'):
            result = self.ff_final(result)

        return result


class Positionwise_FeedForward(nn.Module):
    def __init__(self, sizes, dropout=None,ff_activate='relu'):
        super().__init__()
        self.num_layers = len(sizes)-1
        for i in range(self.num_layers):
            setattr(self, 'w' + str(i), nn.Linear(sizes[i], sizes[i + 1]))

        if dropout == None:
            dropout = collections.defaultdict(int)

        self.dropout = MyDropout(dropout['ff'])
        self.dropout_2 = MyDropout(dropout['ff_2'])
        if ff_activate == 'relu':
            self.activate = nn.ReLU(inplace=True)
        elif ff_activate == 'leaky':
            self.activate = nn.LeakyReLU(inplace=True)

    def forward(self, inp):
        output = inp
        for i in range(self.num_layers):
            if i != 0:
                output = self.activate(output)
            w = getattr(self, 'w' + str(i))
            output = w(output)
            if i == 0:
                output = self.dropout(output)
            if i == 1:
                output = self.dropout_2(output)

        return output



class Transformer_Encoder_Layer(nn.Module):
    def __init__(self, hidden_size, num_heads,
                 relative_position, learnable_position, add_position,
                 layer_preprocess_sequence, layer_postprocess_sequence,
                 dropout=None, scaled=True, ff_size=-1,mode=collections.defaultdict(bool),
                 max_seq_len=-1,pe=None,
                 pe_ss=None, pe_se=None, pe_es=None, pe_ee=None,
                 dvc=None,
                 k_proj=True,q_proj=True,v_proj=True,r_proj=True,
                 attn_ff=True,ff_activate='relu',lattice=False,
                 four_pos_shared=True,four_pos_fusion=None,four_pos_fusion_embedding=None
                 ):
        super().__init__()
        self.four_pos_fusion_embedding=four_pos_fusion_embedding
        self.four_pos_shared=four_pos_shared
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        self.lattice = lattice
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.relative_position = relative_position
        if self.relative_position and self.lattice:
            assert four_pos_fusion is not None
        self.four_pos_fusion = four_pos_fusion
        self.learnable_position = learnable_position
        self.add_position = add_position
        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        self.scaled = scaled
        self.mode = mode
        self.attn_ff = attn_ff
        self.ff_activate = ff_activate

        if self.relative_position and max_seq_len < 0:
            print_info('max_seq_len should be set if relative position encode')
            exit(1208)

        self.max_seq_len = max_seq_len
        if dvc is None:
            dvc = torch.device('cpu')
        self.dvc = dvc

        self.k_proj=k_proj
        self.q_proj=q_proj
        self.v_proj=v_proj
        self.r_proj=r_proj
        if self.relative_position:
            if pe is None:
                pe = get_embedding(max_seq_len,hidden_size,rel_pos_init=self.rel_pos_init)
                pe_sum = pe.sum(dim=-1,keepdim=True)
                if self.pos_norm:
                    with torch.no_grad():
                        pe = pe/pe_sum
                self.pe = nn.Parameter(pe, requires_grad=self.learnable_position)
                if self.four_pos_shared:
                    self.pe_ss = self.pe
                    self.pe_se = self.pe
                    self.pe_es = self.pe
                    self.pe_ee = self.pe
                else:
                    self.pe_ss = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
                    self.pe_se = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
                    self.pe_es = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
                    self.pe_ee = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
            else:
                self.pe = pe
                self.pe_ss = pe_ss
                self.pe_se = pe_se
                self.pe_es = pe_es
                self.pe_ee = pe_ee
        if self.four_pos_fusion_embedding is None:
            self.four_pos_fusion_embedding = \
                Four_Pos_Fusion_Embedding(self.pe,self.four_pos_fusion,self.pe_ss,self.pe_se,self.pe_es,self.pe_ee,
                                          self.max_seq_len,self.hidden_size,self.mode)


        # if self.relative_position:
        #     print('现在还不支持相对编码！')
        #     exit(1208)

        # if not self.add_position:
        #     print_info('现在还不支持位置编码通过concat的方式加入')
        #     exit(1208)

        if dropout == None:
            dropout = collections.defaultdict(int)
        self.dropout = dropout

        if ff_size == -1:
            ff_size = hidden_size
        self.ff_size = ff_size
        # print('dropout:{}'.format(self.dropout))
        self.layer_preprocess = Layer_Process(self.layer_preprocess_sequence,self.hidden_size,self.dropout['pre'])
        self.layer_postprocess = Layer_Process(self.layer_postprocess_sequence,self.hidden_size,self.dropout['post'])
        if self.relative_position:
            if not self.lattice:
                self.attn = MultiHead_Attention_rel(self.hidden_size, self.num_heads,
                                                    pe=self.pe,
                                                    scaled=self.scaled,
                                                    mode=self.mode,
                                                    max_seq_len=self.max_seq_len,
                                                    dvc=self.dvc,
                                                    k_proj=self.k_proj,
                                                    q_proj=self.q_proj,
                                                    v_proj=self.v_proj,
                                                    r_proj=self.r_proj,
                                                    attn_dropout=self.dropout['attn'],
                                                    ff_final=self.attn_ff)
            else:
                self.attn = MultiHead_Attention_Lattice_rel_save_gpumm(self.hidden_size, self.num_heads,
                                                    pe=self.pe,
                                                    pe_ss=self.pe_ss,
                                                    pe_se=self.pe_se,
                                                    pe_es=self.pe_es,
                                                    pe_ee=self.pe_ee,
                                                    scaled=self.scaled,
                                                    mode=self.mode,
                                                    max_seq_len=self.max_seq_len,
                                                    dvc=self.dvc,
                                                    k_proj=self.k_proj,
                                                    q_proj=self.q_proj,
                                                    v_proj=self.v_proj,
                                                    r_proj=self.r_proj,
                                                    attn_dropout=self.dropout['attn'],
                                                    ff_final=self.attn_ff,
                                                    four_pos_fusion=self.four_pos_fusion)

        else:
            self.attn = MultiHead_Attention(self.hidden_size, self.num_heads, self.scaled, mode=self.mode,
                                            k_proj=self.k_proj,q_proj=self.q_proj,v_proj=self.v_proj,
                                            attn_dropout=self.dropout['attn'],
                                            ff_final=self.attn_ff)



        self.ff = Positionwise_FeedForward([hidden_size, ff_size, hidden_size], self.dropout,ff_activate=self.ff_activate)

    def forward(self, inp, seq_len, lex_num=0,pos_s=None,pos_e=None,rel_pos_embedding=None):
        output = inp
        output = self.layer_preprocess(output)
        if self.lattice:
            if self.relative_position:
                if rel_pos_embedding is None:
                    rel_pos_embedding = self.four_pos_fusion_embedding(pos_s,pos_e)
                output = self.attn(output, output, output, seq_len, pos_s=pos_s, pos_e=pos_e, lex_num=lex_num,
                                   rel_pos_embedding=rel_pos_embedding)
            else:
                output = self.attn(output, output, output, seq_len, lex_num)
        else:
            output = self.attn(output, output, output, seq_len)
        output = self.layer_postprocess(output)
        output = self.layer_preprocess(output)
        output = self.ff(output)
        output = self.layer_postprocess(output)

        return output


class Layer_Process(nn.Module):
    def __init__(self, process_sequence, hidden_size, dropout=0, ):
        super().__init__()
        self.process_sequence = process_sequence.lower()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        if 'd' in self.process_sequence:
            self.dropout = MyDropout(dropout)
        if 'n' in self.process_sequence:
            self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, inp):
        output = inp
        for op in self.process_sequence:
            if op == 'a':
                output = output + inp
            elif op == 'd':
                output = self.dropout(output)
            elif op == 'n':
                output = self.layer_norm(output)

        return output




class Transformer_Encoder(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers,
                 relative_position, learnable_position, add_position,
                 layer_preprocess_sequence, layer_postprocess_sequence,
                 dropout=None, scaled=True, ff_size=-1,
                 mode=collections.defaultdict(bool),dvc=None,max_seq_len=-1,pe=None,
                 pe_ss=None,pe_se=None,pe_es=None,pe_ee=None,
                 k_proj=True,q_proj=True,v_proj=True,r_proj=True,
                 attn_ff=True,ff_activate='relu',lattice=False,
                 four_pos_shared=True,four_pos_fusion=None,four_pos_fusion_shared=True):
        '''

        :param input_size:
        :param hidden_size:
        :param num_layers:
        :param relative_position: bool
        :param learnable_position: bool
        :param add_position: bool, if False, concat
        :param layer_preprocess:
        :param layer_postprocess:
        '''
        super().__init__()
        self.four_pos_fusion_shared=four_pos_fusion_shared
        self.four_pos_shared = four_pos_shared
        self.four_pos_fusion = four_pos_fusion
        self.pe = pe
        self.pe_ss = pe_ss
        self.pe_se = pe_se
        self.pe_es = pe_es
        self.pe_ee = pe_ee
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        if self.four_pos_fusion_shared:
            self.four_pos_fusion_embedding = \
                Four_Pos_Fusion_Embedding(self.pe,self.four_pos_fusion,self.pe_ss,self.pe_se,self.pe_es,self.pe_ee,
                                          self.max_seq_len,self.hidden_size,self.mode)
        else:
            self.four_pos_fusion_embedding = None

        self.lattice = lattice
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.relative_position = relative_position
        if self.relative_position and self.lattice:
            assert four_pos_fusion is not None
        self.learnable_position = learnable_position
        self.add_position = add_position
        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        self.scaled = scaled
        self.k_proj=k_proj
        self.q_proj=q_proj
        self.v_proj=v_proj
        self.r_proj=r_proj
        self.attn_ff = attn_ff
        self.ff_activate = ff_activate

        if dvc is None:
            dvc = torch.device('cpu')
        self.dvc = dvc

        if self.relative_position and max_seq_len < 0:
            print_info('max_seq_len should be set if relative position encode')
            exit(1208)

        # if self.relative_position:
        #     print('现在还不支持相对编码！')
        #     exit(1208)

        # if not self.add_position:
        #     print('现在还不支持位置编码通过concat的方式加入')
        #     exit(1208)

        if dropout == None:
            dropout = collections.defaultdict(int)
        self.dropout = dropout

        if ff_size == -1:
            ff_size = hidden_size
        self.ff_size = ff_size

        for i in range(self.num_layers):
            setattr(self, 'layer_{}'.format(i),Transformer_Encoder_Layer(hidden_size, num_heads,
                                                    relative_position, learnable_position, add_position,
                                                    layer_preprocess_sequence, layer_postprocess_sequence,
                                                    dropout,scaled,ff_size,
                                                    mode=self.mode,
                                                    max_seq_len=self.max_seq_len,
                                                    pe=self.pe,
                                                    pe_ss=self.pe_ss,
                                                    pe_se=self.pe_se,
                                                    pe_es=self.pe_es,
                                                    pe_ee=self.pe_ee,
                                                    k_proj=self.k_proj,
                                                    q_proj=self.q_proj,
                                                    v_proj=self.v_proj,
                                                    r_proj=self.r_proj,
                                                    attn_ff=self.attn_ff,
                                                    ff_activate=self.ff_activate,
                                                    lattice=self.lattice,
                                                    four_pos_shared=self.four_pos_shared,
                                                    four_pos_fusion=self.four_pos_fusion,
                                                    four_pos_fusion_embedding=self.four_pos_fusion_embedding

                                                    ))

        self.layer_preprocess = Layer_Process(self.layer_preprocess_sequence,self.hidden_size)

    def forward(self, inp, seq_len,lex_num=0,pos_s=None,pos_e=None):
        output = inp
        if self.relative_position:
            if self.four_pos_fusion_shared and self.lattice:
                rel_pos_embedding = self.four_pos_fusion_embedding(pos_s,pos_e)
            else:
                rel_pos_embedding = None
        else:
            rel_pos_embedding = None
        for i in range(self.num_layers):
            now_layer = getattr(self,'layer_{}'.format(i))
            output = now_layer(output,seq_len,lex_num=lex_num,pos_s=pos_s,pos_e=pos_e,
                               rel_pos_embedding=rel_pos_embedding)

        output = self.layer_preprocess(output)

        return output


class Lattice_Transformer_SeqLabel(nn.Module):
    def __init__(self,lattice_embed, bigram_embed, hidden_size, label_size,
                 num_heads, num_layers,
                 use_abs_pos,use_rel_pos, learnable_position,add_position,
                 layer_preprocess_sequence, layer_postprocess_sequence,
                 ff_size=-1, scaled=True , dropout=None,use_bigram=True,mode=collections.defaultdict(bool),
                 dvc=None,vocabs=None,
                 rel_pos_shared=True,max_seq_len=-1,k_proj=True,q_proj=True,v_proj=True,r_proj=True,
                 self_supervised=False,attn_ff=True,pos_norm=False,ff_activate='relu',rel_pos_init=0,
                 abs_pos_fusion_func='concat',embed_dropout_pos='0',
                 four_pos_shared=True,four_pos_fusion=None,four_pos_fusion_shared=True,bert_embedding=None):
        '''
        :param rel_pos_init: 如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
        如果是1，那么就按-max_len,max_len来初始化

        :param embed_dropout_pos: 如果是0，就直接在embed后dropout，是1就在embed变成hidden size之后再dropout，
        是2就在绝对位置加上之后dropout
        '''
        super().__init__()

        self.use_bert = False
        if bert_embedding is not None:
            self.use_bert = True
            self.bert_embedding = bert_embedding

        self.four_pos_fusion_shared = four_pos_fusion_shared
        self.mode = mode
        self.four_pos_shared = four_pos_shared
        self.abs_pos_fusion_func = abs_pos_fusion_func
        self.lattice_embed = lattice_embed
        self.bigram_embed = bigram_embed
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        # self.relative_position = relative_position
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert four_pos_fusion is not None
        self.four_pos_fusion = four_pos_fusion
        self.learnable_position = learnable_position
        self.add_position = add_position
        self.rel_pos_shared = rel_pos_shared
        self.self_supervised=self_supervised
        self.vocabs = vocabs
        self.attn_ff = attn_ff
        self.pos_norm = pos_norm
        self.ff_activate = ff_activate
        self.rel_pos_init = rel_pos_init
        self.embed_dropout_pos = embed_dropout_pos

        if self.use_rel_pos and max_seq_len < 0:
            print_info('max_seq_len should be set if relative position encode')
            exit(1208)

        self.max_seq_len = max_seq_len

        self.k_proj = k_proj
        self.q_proj = q_proj
        self.v_proj = v_proj
        self.r_proj = r_proj

        self.pe = None

        if self.use_abs_pos:
            self.abs_pos_encode = Absolute_SE_Position_Embedding(self.abs_pos_fusion_func,
                                        self.hidden_size,learnable=self.learnable_position,mode=self.mode,
                                        pos_norm=self.pos_norm)

        if self.use_rel_pos:
            pe = get_embedding(max_seq_len,hidden_size,rel_pos_init=self.rel_pos_init)
            pe_sum = pe.sum(dim=-1,keepdim=True)
            if self.pos_norm:
                with torch.no_grad():
                    pe = pe/pe_sum
            self.pe = nn.Parameter(pe, requires_grad=self.learnable_position)
            if self.four_pos_shared:
                self.pe_ss = self.pe
                self.pe_se = self.pe
                self.pe_es = self.pe
                self.pe_ee = self.pe
            else:
                self.pe_ss = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
                self.pe_se = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
                self.pe_es = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
                self.pe_ee = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
        else:
            self.pe = None
            self.pe_ss = None
            self.pe_se = None
            self.pe_es = None
            self.pe_ee = None

        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        if ff_size==-1:
            ff_size = self.hidden_size
        self.ff_size = ff_size
        self.scaled = scaled
        if dvc == None:
            dvc = 'cpu'
        self.dvc = torch.device(dvc)
        if dropout is None:
            self.dropout = collections.defaultdict(int)
        else:
            self.dropout = dropout
        self.use_bigram = use_bigram

        if self.use_bigram:
            self.bigram_size = self.bigram_embed.embedding.weight.size(1)
            self.char_input_size = self.lattice_embed.embedding.weight.size(1)+self.bigram_embed.embedding.weight.size(1)
        else:
            self.char_input_size = self.lattice_embed.embedding.weight.size(1)

        if self.use_bert:
            self.char_input_size+=self.bert_embedding._embed_size

        self.lex_input_size = self.lattice_embed.embedding.weight.size(1)

        self.embed_dropout = MyDropout(self.dropout['embed'])
        self.gaz_dropout = MyDropout(self.dropout['gaz'])

        self.char_proj = nn.Linear(self.char_input_size,self.hidden_size)
        self.lex_proj = nn.Linear(self.lex_input_size,self.hidden_size)

        self.encoder = Transformer_Encoder(self.hidden_size,self.num_heads,self.num_layers,
                                           relative_position=self.use_rel_pos,
                                           learnable_position=self.learnable_position,
                                           add_position=self.add_position,
                                           layer_preprocess_sequence=self.layer_preprocess_sequence,
                                           layer_postprocess_sequence=self.layer_postprocess_sequence,
                                           dropout=self.dropout,
                                           scaled=self.scaled,
                                           ff_size=self.ff_size,
                                           mode=self.mode,
                                           dvc=self.dvc,
                                           max_seq_len=self.max_seq_len,
                                           pe=self.pe,
                                           pe_ss=self.pe_ss,
                                           pe_se=self.pe_se,
                                           pe_es=self.pe_es,
                                           pe_ee=self.pe_ee,
                                           k_proj=self.k_proj,
                                           q_proj=self.q_proj,
                                           v_proj=self.v_proj,
                                           r_proj=self.r_proj,
                                           attn_ff=self.attn_ff,
                                           ff_activate=self.ff_activate,
                                           lattice=True,
                                           four_pos_fusion=self.four_pos_fusion,
                                           four_pos_fusion_shared=self.four_pos_fusion_shared)

        self.output_dropout = MyDropout(self.dropout['output'])

        self.output = nn.Linear(self.hidden_size,self.label_size)
        if self.self_supervised:
            self.output_self_supervised = nn.Linear(self.hidden_size,len(vocabs['char']))
            print('self.output_self_supervised:{}'.format(self.output_self_supervised.weight.size()))
        self.crf = get_crf_zero_init(self.label_size)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)


    def forward(self, lattice, bigrams, seq_len, lex_num, pos_s, pos_e,
                target, chars_target=None):
        # if self.mode['debug']:
        # print('lattice:{}'.format(lattice))
        # print('bigrams:{}'.format(bigrams))
        # print('seq_len:{}'.format(seq_len))
        # print('lex_num:{}'.format(lex_num))
        # print('pos_s:{}'.format(pos_s))
        # print('pos_e:{}'.format(pos_e))

        batch_size = lattice.size(0)
        max_seq_len_and_lex_num = lattice.size(1)
        max_seq_len = bigrams.size(1)

        raw_embed = self.lattice_embed(lattice)
        #raw_embed 是字和词的pretrain的embedding，但是是分别trian的，所以需要区分对待
        if self.use_bigram:
            bigrams_embed = self.bigram_embed(bigrams)
            bigrams_embed = torch.cat([bigrams_embed,
                                       torch.zeros(size=[batch_size,max_seq_len_and_lex_num-max_seq_len,
                                                         self.bigram_size]).to(bigrams_embed)],dim=1)
            raw_embed_char = torch.cat([raw_embed, bigrams_embed],dim=-1)
        else:
            raw_embed_char = raw_embed
        # print('raw_embed_char_1:{}'.format(raw_embed_char[:1,:3,-5:]))

        if self.use_bert:
            bert_pad_length = lattice.size(1)-max_seq_len
            char_for_bert = lattice[:, :max_seq_len]
            mask = seq_len_to_mask(seq_len).bool()
            char_for_bert = char_for_bert.masked_fill((~mask),self.vocabs['lattice'].padding_idx)
            bert_embed = self.bert_embedding(char_for_bert)
            bert_embed = torch.cat([bert_embed,
                                    torch.zeros(size=[batch_size,bert_pad_length,bert_embed.size(-1)],
                                                device = bert_embed.device,
                                                requires_grad=False)],dim=-2)
            # print('bert_embed:{}'.format(bert_embed[:1, :3, -5:]))
            raw_embed_char = torch.cat([raw_embed_char, bert_embed],dim=-1)

        # print('raw_embed_char:{}'.format(raw_embed_char[:1,:3,-5:]))

        if self.embed_dropout_pos == '0':
            raw_embed_char = self.embed_dropout(raw_embed_char)
            raw_embed = self.gaz_dropout(raw_embed)

        # print('raw_embed_char_dp:{}'.format(raw_embed_char[:1,:3,-5:]))



        embed_char = self.char_proj(raw_embed_char)
        # print('char_proj:',list(self.char_proj.parameters())[0].data[:2][:2])
        # print('embed_char_:{}'.format(embed_char[:1,:3,:4]))



        if self.mode['debug']:
            print('embed_char:{}'.format(embed_char[:2]))
        char_mask = seq_len_to_mask(seq_len,max_len=max_seq_len_and_lex_num).bool()
        # if self.embed_dropout_pos == '1':
        #     embed_char = self.embed_dropout(embed_char)
        embed_char.masked_fill_(~(char_mask.unsqueeze(-1)), 0)

        embed_lex = self.lex_proj(raw_embed)
        if self.mode['debug']:
            print('embed_lex:{}'.format(embed_lex[:2]))
        # if self.embed_dropout_pos == '1':
        #     embed_lex = self.embed_dropout(embed_lex)

        lex_mask = (seq_len_to_mask(seq_len+lex_num).bool() ^ char_mask.bool())
        embed_lex.masked_fill_(~(lex_mask).unsqueeze(-1), 0)

        assert char_mask.size(1) == lex_mask.size(1)
        # print('embed_char:{}'.format(embed_char[:1,:3,:4]))
        # print('embed_lex:{}'.format(embed_lex[:1,:3,:4]))




        embedding = embed_char + embed_lex
        if self.mode['debug']:
            print('embedding:{}'.format(embedding[:2]))

        if self.embed_dropout_pos == '1':
            embedding = self.embed_dropout(embedding)

        if self.use_abs_pos:
            embedding = self.abs_pos_encode(embedding,pos_s,pos_e)

        if self.embed_dropout_pos == '2':
            embedding = self.embed_dropout(embedding)
        # embedding = self.embed_dropout(embedding)
        # print('*1*')
        # print(embedding.size())
        # print('merged_embedding:{}'.format(embedding[:1,:3,:4]))
        # exit()
        encoded = self.encoder(embedding,seq_len,lex_num=lex_num,pos_s=pos_s,pos_e=pos_e)

        if hasattr(self,'output_dropout'):
            encoded = self.output_dropout(encoded)


        encoded = encoded[:,:max_seq_len,:]
        logits = self.output(encoded)

        mask = seq_len_to_mask(seq_len).bool()


        loss = self.crf(logits, target, mask).mean(dim=0)
        if self.training:
            return {'loss': loss}


        # if self.self_supervised:
        #     chars_pred = self.output_self_supervised(encoded)
        #     chars_pred = chars_pred.view(size=[batch_size*max_seq_len,-1])
        #     chars_target = chars_target.view(size=[batch_size*max_seq_len])
        #     self_supervised_loss = self.loss_func(chars_pred,chars_target)
        #     loss += self_supervised_loss


        # return {'loss': loss}
        # else:

        pred, path = self.crf.viterbi_decode(logits, mask)
        result = {'pred': pred}
        if self.self_supervised:
            chars_pred = self.output_self_supervised(encoded)
            result['chars_pred'] = chars_pred

        result['trans_m'] = self.crf.trans_m.data
        result['loss'] = loss
        result['logits'] = logits
        result['mask'] = mask
        return result


class StaticEmbedding(TokenEmbedding):
    """
    StaticEmbedding组件. 给定预训练embedding的名称或路径，根据vocab从embedding中抽取相应的数据(只会将出现在vocab中的词抽取出来，
    如果没有找到，则会随机初始化一个值(但如果该word是被标记为no_create_entry的话，则不会单独创建一个值，而是会被指向unk的index))。
    当前支持自动下载的预训练vector有以下的几种(待补充);

    Example::

        >>> from fastNLP import Vocabulary
        >>> from fastNLP.embeddings import StaticEmbedding
        >>> vocab = Vocabulary().add_word_lst("The whether is good .".split())
        >>> embed = StaticEmbedding(vocab, model_dir_or_name='en-glove-50d')

        >>> vocab = Vocabulary().add_word_lst(["The", 'the', "THE"])
        >>> embed = StaticEmbedding(vocab, model_dir_or_name="en-glove-50d", lower=True)
        >>> # "the", "The", "THE"它们共用一个vector，且将使用"the"在预训练词表中寻找它们的初始化表示。

        >>> vocab = Vocabulary().add_word_lst(["The", "the", "THE"])
        >>> embed = StaticEmbedding(vocab, model_dir_or_name=None, embedding_dim=5, lower=True)
        >>> words = torch.LongTensor([[vocab.to_index(word) for word in ["The", "the", "THE"]]])
        >>> embed(words)
        >>> tensor([[[ 0.5773,  0.7251, -0.3104,  0.0777,  0.4849],
                     [ 0.5773,  0.7251, -0.3104,  0.0777,  0.4849],
                     [ 0.5773,  0.7251, -0.3104,  0.0777,  0.4849]]],
                   grad_fn=<EmbeddingBackward>)  # 每种word的输出是一致的。

    """

    def __init__(self, vocab: Vocabulary, model_dir_or_name: str = 'en', embedding_dim=-1, requires_grad: bool = True,
                 init_method=None, lower=False, dropout=0, word_dropout=0, normalize=False, min_freq=1, **kwargs):
        """

        :param vocab: Vocabulary. 若该项为None则会读取所有的embedding。
        :param model_dir_or_name: 可以有两种方式调用预训练好的static embedding：第一种是传入embedding文件夹(文件夹下应该只有一个
            以.txt作为后缀的文件)或文件路径；第二种是传入embedding的名称，第二种情况将自动查看缓存中是否存在该模型，没有的话将自动下载。
            如果输入为None则使用embedding_dim的维度随机初始化一个embedding。
        :param int embedding_dim: 随机初始化的embedding的维度，当该值为大于0的值时，将忽略model_dir_or_name。
        :param bool requires_grad: 是否需要gradient. 默认为True
        :param callable init_method: 如何初始化没有找到的值。可以使用torch.nn.init.*中各种方法, 传入的方法应该接受一个tensor，并
            inplace地修改其值。
        :param bool lower: 是否将vocab中的词语小写后再和预训练的词表进行匹配。如果你的词表中包含大写的词语，或者就是需要单独
            为大写的词语开辟一个vector表示，则将lower设置为False。
        :param float dropout: 以多大的概率对embedding的表示进行Dropout。0.1即随机将10%的值置为0。
        :param float word_dropout: 以多大的概率将一个词替换为unk。这样既可以训练unk也是一定的regularize。
        :param bool normalize: 是否对vector进行normalize，使得每个vector的norm为1。
        :param int min_freq: Vocabulary词频数小于这个数量的word将被指向unk。
        :param dict kwarngs: only_train_min_freq, 仅对train中的词语使用min_freq筛选; only_norm_found_vector是否仅对在预训练中找到的词语使用normalize。
        """
        super(StaticEmbedding, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)
        if embedding_dim > 0:
            model_dir_or_name = None

        # 得到cache_path
        if model_dir_or_name is None:
            assert embedding_dim >= 1, "The dimension of embedding should be larger than 1."
            embedding_dim = int(embedding_dim)
            model_path = None
        elif model_dir_or_name.lower() in PRETRAIN_STATIC_FILES:
            model_url = _get_embedding_url('static', model_dir_or_name.lower())
            model_path = cached_path(model_url, name='embedding')
            # 检查是否存在
        elif os.path.isfile(os.path.abspath(os.path.expanduser(model_dir_or_name))):
            model_path = os.path.abspath(os.path.expanduser(model_dir_or_name))
        elif os.path.isdir(os.path.abspath(os.path.expanduser(model_dir_or_name))):
            model_path = _get_file_name_base_on_postfix(os.path.abspath(os.path.expanduser(model_dir_or_name)), '.txt')
        else:
            raise ValueError(f"Cannot recognize {model_dir_or_name}.")

        # 根据min_freq缩小vocab
        truncate_vocab = (vocab.min_freq is None and min_freq > 1) or (vocab.min_freq and vocab.min_freq < min_freq)
        if truncate_vocab:
            truncated_vocab = deepcopy(vocab)
            truncated_vocab.min_freq = min_freq
            truncated_vocab.word2idx = None
            if lower:  # 如果有lower，将大小写的的freq需要同时考虑到
                lowered_word_count = defaultdict(int)
                for word, count in truncated_vocab.word_count.items():
                    lowered_word_count[word.lower()] += count
                for word in truncated_vocab.word_count.keys():
                    word_count = truncated_vocab.word_count[word]
                    if lowered_word_count[word.lower()] >= min_freq and word_count < min_freq:
                        truncated_vocab.add_word_lst([word] * (min_freq - word_count),
                                                     no_create_entry=truncated_vocab._is_word_no_create_entry(word))

            # 只限制在train里面的词语使用min_freq筛选
            if kwargs.get('only_train_min_freq', False) and model_dir_or_name is not None:
                for word in truncated_vocab.word_count.keys():
                    if truncated_vocab._is_word_no_create_entry(word) and truncated_vocab.word_count[word] < min_freq:
                        truncated_vocab.add_word_lst([word] * (min_freq - truncated_vocab.word_count[word]),
                                                     no_create_entry=True)
            truncated_vocab.build_vocab()
            truncated_words_to_words = torch.arange(len(vocab)).long()
            for word, index in vocab:
                truncated_words_to_words[index] = truncated_vocab.to_index(word)
            logger.info(
                f"{len(vocab) - len(truncated_vocab)} out of {len(vocab)} words have frequency less than {min_freq}.")
            vocab = truncated_vocab

        self.only_norm_found_vector = kwargs.get('only_norm_found_vector', False)
        # 读取embedding
        if lower:
            lowered_vocab = Vocabulary(padding=vocab.padding, unknown=vocab.unknown)
            for word, index in vocab:
                if vocab._is_word_no_create_entry(word):
                    lowered_vocab.add_word(word.lower(), no_create_entry=True)
                else:
                    lowered_vocab.add_word(word.lower())  # 先加入需要创建entry的
            logger.info(f"All word in the vocab have been lowered. There are {len(vocab)} words, {len(lowered_vocab)} "
                        f"unique lowered words.")
            if model_path:
                embedding = self._load_with_vocab(model_path, vocab=lowered_vocab, init_method=init_method)
            else:
                embedding = self._randomly_init_embed(len(vocab), embedding_dim, init_method)
                self.register_buffer('words_to_words', torch.arange(len(vocab)).long())
            if lowered_vocab.unknown:
                unknown_idx = lowered_vocab.unknown_idx
            else:
                unknown_idx = embedding.size(0) - 1  # 否则是最后一个为unknow
                self.register_buffer('words_to_words', torch.arange(len(vocab)).long())
            words_to_words = torch.full((len(vocab),), fill_value=unknown_idx).long()
            for word, index in vocab:
                if word not in lowered_vocab:
                    word = word.lower()
                    if word not in lowered_vocab and lowered_vocab._is_word_no_create_entry(word):
                        continue  # 如果不需要创建entry,已经默认unknown了
                words_to_words[index] = self.words_to_words[lowered_vocab.to_index(word)]
            self.register_buffer('words_to_words', words_to_words)
            self._word_unk_index = lowered_vocab.unknown_idx  # 替换一下unknown的index
        else:
            if model_path:
                embedding = self._load_with_vocab(model_path, vocab=vocab, init_method=init_method)
            else:
                embedding = self._randomly_init_embed(len(vocab), embedding_dim, init_method)
                self.register_buffer('words_to_words', torch.arange(len(vocab)).long())
        if not self.only_norm_found_vector and normalize:
            embedding /= (torch.norm(embedding, dim=1, keepdim=True) + 1e-12)

        if truncate_vocab:
            for i in range(len(truncated_words_to_words)):
                index_in_truncated_vocab = truncated_words_to_words[i]
                truncated_words_to_words[i] = self.words_to_words[index_in_truncated_vocab]
            del self.words_to_words
            self.register_buffer('words_to_words', truncated_words_to_words)
        self.embedding = nn.Embedding(num_embeddings=embedding.shape[0], embedding_dim=embedding.shape[1],
                                      padding_idx=vocab.padding_idx,
                                      max_norm=None, norm_type=2, scale_grad_by_freq=False,
                                      sparse=False, _weight=embedding)
        self._embed_size = self.embedding.weight.size(1)
        self.requires_grad = requires_grad
        self.dropout = MyDropout(dropout)

    def _randomly_init_embed(self, num_embedding, embedding_dim, init_embed=None):
        """

        :param int num_embedding: embedding的entry的数量
        :param int embedding_dim: embedding的维度大小
        :param callable init_embed: 初始化方法
        :return: torch.FloatTensor
        """
        embed = torch.zeros(num_embedding, embedding_dim)

        if init_embed is None:
            nn.init.uniform_(embed, -np.sqrt(3 / embedding_dim), np.sqrt(3 / embedding_dim))
        else:
            init_embed(embed)

        return embed

    def _load_with_vocab(self, embed_filepath, vocab, dtype=np.float32, padding='<pad>', unknown='<unk>',
                         error='ignore', init_method=None):
        """
        从embed_filepath这个预训练的词向量中抽取出vocab这个词表的词的embedding。EmbedLoader将自动判断embed_filepath是
        word2vec(第一行只有两个元素)还是glove格式的数据。

        :param str embed_filepath: 预训练的embedding的路径。
        :param vocab: 词表 :class:`~fastNLP.Vocabulary` 类型，读取出现在vocab中的词的embedding。
            没有出现在vocab中的词的embedding将通过找到的词的embedding的正态分布采样出来，以使得整个Embedding是同分布的。
        :param dtype: 读出的embedding的类型
        :param str padding: 词表中padding的token
        :param str unknown: 词表中unknown的token
        :param str error: `ignore` , `strict` ; 如果 `ignore` ，错误将自动跳过; 如果 `strict` , 错误将抛出。
            这里主要可能出错的地方在于词表有空行或者词表出现了维度不一致。
        :param init_method: 如何初始化没有找到的值。可以使用torch.nn.init.*中各种方法。默认使用torch.nn.init.zeros_
        :return torch.tensor:  shape为 [len(vocab), dimension], dimension由pretrain的embedding决定。
        """
        assert isinstance(vocab, Vocabulary), "Only fastNLP.Vocabulary is supported."
        if not os.path.exists(embed_filepath):
            raise FileNotFoundError("`{}` does not exist.".format(embed_filepath))
        with open(embed_filepath, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            parts = line.split()
            start_idx = 0
            if len(parts) == 2:
                dim = int(parts[1])
                start_idx += 1
            else:
                dim = len(parts) - 1
                f.seek(0)
            matrix = {}
            if vocab.padding:
                matrix[vocab.padding_idx] = torch.zeros(dim)
            if vocab.unknown:
                matrix[vocab.unknown_idx] = torch.zeros(dim)
            found_count = 0
            found_unknown = False
            for idx, line in enumerate(f, start_idx):
                try:
                    parts = line.strip().split()
                    word = ''.join(parts[:-dim])
                    nums = parts[-dim:]
                    # 对齐unk与pad
                    if word == padding and vocab.padding is not None:
                        word = vocab.padding
                    elif word == unknown and vocab.unknown is not None:
                        word = vocab.unknown
                        found_unknown = True
                    if word in vocab:
                        index = vocab.to_index(word)
                        matrix[index] = torch.from_numpy(np.fromstring(' '.join(nums), sep=' ', dtype=dtype, count=dim))
                        if self.only_norm_found_vector:
                            matrix[index] = matrix[index] / np.linalg.norm(matrix[index])
                        found_count += 1
                except Exception as e:
                    if error == 'ignore':
                        warnings.warn("Error occurred at the {} line.".format(idx))
                    else:
                        logger.error("Error occurred at the {} line.".format(idx))
                        raise e
            logger.info("Found {} out of {} words in the pre-training embedding.".format(found_count, len(vocab)))
            for word, index in vocab:
                if index not in matrix and not vocab._is_word_no_create_entry(word):
                    if found_unknown:  # 如果有unkonwn，用unknown初始化
                        matrix[index] = matrix[vocab.unknown_idx]
                    else:
                        matrix[index] = None
            # matrix中代表是需要建立entry的词
            vectors = self._randomly_init_embed(len(matrix), dim, init_method)

            if vocab.unknown is None:  # 创建一个专门的unknown
                unknown_idx = len(matrix)
                vectors = torch.cat((vectors, torch.zeros(1, dim)), dim=0).contiguous()
            else:
                unknown_idx = vocab.unknown_idx
            self.register_buffer('words_to_words', torch.full((len(vocab),), fill_value=unknown_idx, dtype=torch.long).long())
            for index, (index_in_vocab, vec) in enumerate(matrix.items()):
                if vec is not None:
                    vectors[index] = vec
                self.words_to_words[index_in_vocab] = index

            return vectors

    def drop_word(self, words):
        """
        按照设定随机将words设置为unknown_index。

        :param torch.LongTensor words: batch_size x max_len
        :return:
        """
        if self.word_dropout > 0 and self.training:
            mask = torch.rand(words.size())
            mask = mask.to(words.device)
            mask = mask.lt(self.word_dropout)
            # mask = torch.full_like(words, fill_value=self.word_dropout, dtype=torch.float)
            #             # mask = torch.bernoulli(mask).eq(1)  # dropout_word越大，越多位置为1
            #             # mask = mask.to(words.device)
            pad_mask = words.ne(self._word_pad_index)
            mask = mask.__and__(pad_mask)
            words = words.masked_fill(mask, self._word_unk_index)
        return words

    def forward(self, words):
        """
        传入words的index

        :param words: torch.LongTensor, [batch_size, max_len]
        :return: torch.FloatTensor, [batch_size, max_len, embed_size]
        """
        if hasattr(self, 'words_to_words'):
            words = self.words_to_words[words]
        words = self.drop_word(words)
        words = self.embedding(words)
        words = self.dropout(words)
        return words


class BertEmbedding(ContextualEmbedding):
    """
    使用BERT对words进行编码的Embedding。建议将输入的words长度限制在430以内，而不要使用512(根据预训练模型参数，可能有变化)。这是由于
    预训练的bert模型长度限制为512个token，而因为输入的word是未进行word piece分割的(word piece的分割有BertEmbedding在输入word
    时切分)，在分割之后长度可能会超过最大长度限制。

    BertEmbedding可以支持自动下载权重，当前支持的模型有以下的几种(待补充):

    Example::

        >>> import torch
        >>> from fastNLP import Vocabulary
        >>> from fastNLP.embeddings import BertEmbedding
        >>> vocab = Vocabulary().add_word_lst("The whether is good .".split())
        >>> embed = BertEmbedding(vocab, model_dir_or_name='en-base-uncased', requires_grad=False, layers='4,-2,-1')
        >>> words = torch.LongTensor([[vocab.to_index(word) for word in "The whether is good .".split()]])
        >>> outputs = embed(words)
        >>> outputs.size()
        >>> # torch.Size([1, 5, 2304])
    """

    def __init__(self, vocab: Vocabulary, model_dir_or_name: str = 'en-base-uncased', layers: str = '-1',
                 pool_method: str = 'first', word_dropout=0, dropout=0, include_cls_sep: bool = False,
                 pooled_cls=True, requires_grad: bool = True, auto_truncate: bool = False):
        """

        :param ~fastNLP.Vocabulary vocab: 词表
        :param str model_dir_or_name: 模型所在目录或者模型的名称。当传入模型所在目录时，目录中应该包含一个词表文件(以.txt作为后缀名),
            权重文件(以.bin作为文件后缀名), 配置文件(以.json作为后缀名)。
        :param str layers: 输出embedding表示来自于哪些层，不同层的结果按照layers中的顺序在最后一维concat起来。以','隔开层数，层的序号是
            从0开始，可以以负数去索引倒数几层。
        :param str pool_method: 因为在bert中，每个word会被表示为多个word pieces, 当获取一个word的表示的时候，怎样从它的word pieces
            中计算得到它对应的表示。支持 ``last`` , ``first`` , ``avg`` , ``max``。
        :param float word_dropout: 以多大的概率将一个词替换为unk。这样既可以训练unk也是一定的regularize。
        :param float dropout: 以多大的概率对embedding的表示进行Dropout。0.1即随机将10%的值置为0。
        :param bool include_cls_sep: bool，在bert计算句子的表示的时候，需要在前面加上[CLS]和[SEP], 是否在结果中保留这两个内容。 这样
            会使得word embedding的结果比输入的结果长两个token。如果该值为True，则在使用 :class::StackEmbedding 可能会与其它类型的
            embedding长度不匹配。
        :param bool pooled_cls: 返回的[CLS]是否使用预训练中的BertPool映射一下，仅在include_cls_sep时有效。如果下游任务只取[CLS]做预测，
            一般该值为True。
        :param bool requires_grad: 是否需要gradient以更新Bert的权重。
        :param bool auto_truncate: 当句子words拆分为word pieces长度超过bert最大允许长度(一般为512), 自动截掉拆分后的超过510个
            word pieces后的内容，并将第512个word piece置为[SEP]。超过长度的部分的encode结果直接全部置零。一般仅有只使用[CLS]
            来进行分类的任务将auto_truncate置为True。
        """
        super(BertEmbedding, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)
        self.device_cpu = torch.device('cpu')
        if model_dir_or_name.lower() in PRETRAINED_BERT_MODEL_DIR:
            if 'cn' in model_dir_or_name.lower() and pool_method not in ('first', 'last'):
                logger.warning("For Chinese bert, pooled_method should choose from 'first', 'last' in order to achieve"
                               " faster speed.")
                warnings.warn("For Chinese bert, pooled_method should choose from 'first', 'last' in order to achieve"
                              " faster speed.")
        self.dropout_p = dropout
        self._word_sep_index = None
        if '[SEP]' in vocab:
            self._word_sep_index = vocab['[SEP]']

        self.model = _WordBertModel(model_dir_or_name=model_dir_or_name, vocab=vocab, layers=layers,
                                    pool_method=pool_method, include_cls_sep=include_cls_sep,
                                    pooled_cls=pooled_cls, auto_truncate=auto_truncate, min_freq=2)

        self.requires_grad = requires_grad
        self._embed_size = len(self.model.layers) * self.model.encoder.hidden_size

    def _delete_model_weights(self):
        del self.model

    def forward(self, words):
        """
        计算words的bert embedding表示。计算之前会在每句话的开始增加[CLS]在结束增加[SEP], 并根据include_cls_sep判断要不要
            删除这两个token的表示。

        :param torch.LongTensor words: [batch_size, max_len]
        :return: torch.FloatTensor. batch_size x max_len x (768*len(self.layers))
        """
        words = self.drop_word(words)
        outputs = self._get_sent_reprs(words)
        if outputs is not None:
            if self.dropout_p >1e-5:
                return self.dropout(outputs)
            else:
                return outputs
        outputs = self.model(words)
        # print(outputs.size())

        outputs = torch.cat([*outputs], dim=-1)
        # print(outputs.size())
        # exit()
        if self.dropout_p > 1e-5:
            return self.dropout(outputs)
        else:
            return outputs

    def drop_word(self, words):
        """
        按照设定随机将words设置为unknown_index。

        :param torch.LongTensor words: batch_size x max_len
        :return:
        """
        if self.word_dropout > 0 and self.training:
            with torch.no_grad():
                if self._word_sep_index:  # 不能drop sep
                    sep_mask = words.eq(self._word_sep_index)

                mask = torch.full(words.size(), fill_value=self.word_dropout, dtype=torch.float)
                # print(mask.device)
                # print(mask)
                # print(mask.device)
                # exit()
                # mask = mask.to(self.device_cpu)
                mask = torch.bernoulli(mask).eq(1)  # dropout_word越大，越多位置为1
                mask = mask.to(words.device)
                pad_mask = words.ne(0)
                mask = pad_mask.__and__(mask)  # pad的位置不为unk
                words = words.masked_fill(mask, self._word_unk_index)
                if self._word_sep_index:
                    words.masked_fill_(sep_mask, self._word_sep_index)
        return words


def get_peking_time():

    tz = pytz.timezone('Asia/Shanghai')  # 东八区

    t = datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone('Asia/Shanghai')).strftime('%Y_%m_%d_%H_%M_%S')
    return t


# @cache_results(_cache_fp='need_to_defined_fp',_refresh=True)
def equip_chinese_ner_with_lexicon(datasets, vocabs, embeddings, w_list,word_embedding_path=None, \
                only_lexicon_in_train=False, word_char_mix_embedding_path=None, number_normalized=False, \
                lattice_min_freq=1, only_train_min_freq=0, with_placeholder=True, with_test_a=False, **kwargs):
    def normalize_char(inp):
        result = []
        for c in inp:
            if c.isdigit():
                result.append('0')
            else:
                result.append(c)

        return result

    def normalize_bigram(inp):
        result = []
        for bi in inp:
            tmp = bi
            if tmp[0].isdigit():
                tmp = '0'+tmp[:1]
            if tmp[1].isdigit():
                tmp = tmp[0]+'0'

            result.append(tmp)
        return result

    if number_normalized == 3:
        for k,v in datasets.items():
            v.apply_field(normalize_char,'chars','chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                no_create_entry_dataset=[datasets['dev'], datasets['test']])

        for k,v in datasets.items():
            v.apply_field(normalize_bigram,'bigrams','bigrams')
        vocabs['bigram'] = Vocabulary()
        vocabs['bigram'].from_dataset(datasets['train'], field_name='bigrams',
                                  no_create_entry_dataset=[datasets['dev'], datasets['test']])


    if only_lexicon_in_train:
        print('已支持只加载在trian中出现过的词汇')

    def get_skip_path(chars, w_trie):
        sentence = ''.join(chars)
        result = w_trie.get_lexicon(sentence)
        # print(result)

        return result
    a = DataSet()
    w_trie = Trie()
    for w in w_list:
        w_trie.insert(w)


    if only_lexicon_in_train:
        lexicon_in_train = set()
        for s in datasets['train']['chars']:
            lexicon_in_s = w_trie.get_lexicon(s)
            for s,e,lexicon in lexicon_in_s:
                lexicon_in_train.add(''.join(lexicon))

        print('lexicon in train:{}'.format(len(lexicon_in_train)))
        print('i.e.: {}'.format(list(lexicon_in_train)[:10]))
        w_trie = Trie()
        for w in lexicon_in_train:
            w_trie.insert(w)

    for k,v in datasets.items():
        v.apply_field(partial(get_skip_path,w_trie=w_trie),'chars','lexicons')
        v.apply_field(copy.copy, 'chars','raw_chars')
        v.add_seq_len('lexicons','lex_num')
        v.apply_field(lambda x:list(map(lambda y: y[0], x)), 'lexicons', 'lex_s')
        v.apply_field(lambda x: list(map(lambda y: y[1], x)), 'lexicons', 'lex_e')


    if number_normalized == 1:
        for k,v in datasets.items():
            v.apply_field(normalize_char,'chars','chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                no_create_entry_dataset=[datasets['dev'], datasets['test']])

    if number_normalized == 2:
        for k,v in datasets.items():
            v.apply_field(normalize_char,'chars','chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                no_create_entry_dataset=[datasets['dev'], datasets['test']])

        for k,v in datasets.items():
            v.apply_field(normalize_bigram,'bigrams','bigrams')
        vocabs['bigram'] = Vocabulary()
        vocabs['bigram'].from_dataset(datasets['train'], field_name='bigrams',
                                  no_create_entry_dataset=[datasets['dev'], datasets['test']])


    def concat(ins):
        chars = ins['chars']
        lexicons = ins['lexicons']
        result = chars + list(map(lambda x:x[2],lexicons))
        return result

    def get_pos_s(ins):
        lex_s = ins['lex_s']
        seq_len = ins['seq_len']
        pos_s = list(range(seq_len)) + lex_s

        return pos_s

    def get_pos_e(ins):
        lex_e = ins['lex_e']
        seq_len = ins['seq_len']
        pos_e = list(range(seq_len)) + lex_e

        return pos_e



    for k,v in datasets.items():
        v.apply(concat,new_field_name='lattice')
        v.set_input('lattice')
        v.apply(get_pos_s,new_field_name='pos_s')
        v.apply(get_pos_e, new_field_name='pos_e')
        v.set_input('pos_s','pos_e')


    word_vocab = Vocabulary()
    word_vocab.add_word_lst(w_list)
    vocabs['word'] = word_vocab

    lattice_vocab = Vocabulary()
    # lattice_vocab.from_dataset(datasets['train'],field_name='lattice',no_create_entry_dataset=[datasets['dev']])
    if with_placeholder is True and with_test_a is False:
        lattice_vocab.from_dataset(datasets['train'],field_name='lattice',no_create_entry_dataset=[datasets['dev'], datasets['placeholder']])
    elif with_placeholder is True and with_test_a is True:
        lattice_vocab.from_dataset(datasets['train'],field_name='lattice',no_create_entry_dataset=[datasets['dev'], datasets['placeholder'], datasets['test_a']])
        print('dataset create with test_a')
    else:
        lattice_vocab.from_dataset(datasets['train'],field_name='lattice',no_create_entry_dataset=[datasets['dev']])
    # lattice_vocab.from_dataset(datasets['train'],field_name='lattice',
                            #    no_create_entry_dataset=[v for k,v in datasets.items() if k != 'train'])
    vocabs['lattice'] = lattice_vocab
    vocabs['lattice'] = lattice_vocab

    if word_embedding_path is not None:
        word_embedding = StaticEmbedding(word_vocab,word_embedding_path,word_dropout=0)
        embeddings['word'] = word_embedding

    if word_char_mix_embedding_path is not None:
        lattice_embedding = StaticEmbedding(lattice_vocab, word_char_mix_embedding_path,word_dropout=0.01,
                                            min_freq=lattice_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['lattice'] = lattice_embedding

    vocabs['char'].index_dataset(* (datasets.values()),
                             field_name='chars', new_field_name='chars')
    vocabs['bigram'].index_dataset(* (datasets.values()),
                               field_name='bigrams', new_field_name='bigrams')
    vocabs['label'].index_dataset(* (datasets.values()),
                              field_name='target', new_field_name='target')
    vocabs['lattice'].index_dataset(* (datasets.values()),
                                    field_name='lattice', new_field_name='lattice')

    return datasets, vocabs, embeddings


def norm_static_embedding(x,norm=1):
    with torch.no_grad():
        x.embedding.weight /= (torch.norm(x.embedding.weight, dim=1, keepdim=True) + 1e-12)
        x.embedding.weight *= norm


# @cache_results(_cache_fp='cache/load_yangjie_rich_pretrain_word_list',_refresh=False)
def load_yangjie_rich_pretrain_word_list(embedding_path,drop_characters=True, **kwargs):
    f = open(embedding_path,'r')
    lines = f.readlines()
    w_list = []
    for line in lines:
        splited = line.strip().split(' ')
        w = splited[0]
        w_list.append(w)

    if drop_characters:
        w_list = list(filter(lambda x:len(x) != 1, w_list))

    return w_list

