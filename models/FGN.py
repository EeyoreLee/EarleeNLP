# -*- encoding: utf-8 -*-
'''
@create_time: 2021/08/16 14:10:21
@author: lichunyu
'''

import torch
from torch.functional import split
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class CGSCNN(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, dropout_prob=0.2):
        super().__init__()
        self.img_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        # self.en_embedding = nn
        self.conv3d = nn.Conv3d(8, (3,3,3), 1)
        self.conv2d = nn.Conv2d(16, (3,3), 1)
        self.pool2d = nn.MaxPool2d((2,2), 2)
        self.conv2d2 = nn.Conv2d(32, (2,2))
        self.pool2d2 = nn.MaxPool2d((2,2))
        self.max_pool1d = nn.MaxPool1d(4)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, char_input, bool_input=None):
        """[summary]

        :param char_input: [description]
        :type char_input: [type]
        :param bool_input: [是否是中文], defaults to None
        :type bool_input: [type], optional
        :return: [description]
        :rtype: [type]
        """
        embed = self.img_embedding(char_input)
        droped_embed = self.dropout(embed)
        output = self.conv3d(droped_embed)
        output = self.conv2d(output)
        output = self.pool2d(output)
        output = self.conv2d2(output)
        output = self.pool2d2(output)
        output = torch.reshape(output, (4, 64))
        output = self.max_pool1d(output)  # TODO 加 关于英文的处理
        return output


def slide_window(_tensor, window_size=1, stride=None, pad=False):
    batch = _tensor.shape[0]
    seq_dim = _tensor.shape[1]
    vec_len = _tensor.shape[-1]
    if stride is None:
        stride = window_size
    if pad is True:
        pad_num = (_tensor.shape[-1] - window_size) % stride
        pad_dim = (
            0,pad_num,
            0,0,
            0,0
        )
        _tensor = F.pad(_tensor, pad_dim)
        vec_len += pad_num
    slice_tensor = []
    for i in range(0, vec_len-window_size+1, stride):
        slice = _tensor[:, :, i:i+window_size].unsqueeze(-2)
        slice_tensor.append(slice)
    tensor_grouped = torch.cat(slice_tensor, dim=-2)
    return tensor_grouped


class OosSlidingWindow(nn.Module):

    def __init__(self, k_c, s_c, k_g, s_g, d_c=512, d_g=64):
        super().__init__()
        self.k_c = k_c
        self.s_c = s_c
        self.k_g = k_g
        self.s_g = s_g
        self.d_c = d_c
        self.d_g = d_g
        assert ((d_c - k_c) / s_c) == (d_g - k_g) / s_g
        self.n = int(((d_c - k_c) / s_c) + 1)

    def forward(self, c_s, g_s):
        batch = c_s.shape[0]
        seq_len = c_s.shape[1]
        c_s_group = slide_window(c_s, window_size=self.k_c, stride=self.s_c)
        g_s_group = slide_window(g_s, window_size=self.k_g, stride=self.s_g)
        c_s_group_r = c_s_group.repeat(1, 1, 1, self.n).reshape(batch, seq_len, self.n**2, -1, c_s_group.shape[-1])
        g_s_group_r = g_s_group.repeat(1, 1, self.n, 1).unsqueeze(-2)
        outer = torch.einsum('bsihc,bsihg->bsicg', [c_s_group_r, g_s_group_r]).squeeze(dim=-3).reshape(batch, \
                seq_len, self.n, self.n, c_s_group_r.shape[-1], g_s_group_r.shape[-1]).flatten(start_dim=-3)
        return outer


class SliceAttention(nn.Module):

    def __init__(self, k_c, k_g):
        super().__init__()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.slice_linear = nn.Linear(k_c*k_g, k_c*k_g)
        self.query = nn.Linear(k_c*k_g, k_c*k_g)

    def forward(self, outer):
        k = self.slice_linear(outer)
        k = self.sigmoid(k)
        k = k.transpose(-1, -2)
        q = self.query(outer)
        q = self.sigmoid(q)
        attn = self.softmax(q*k)
        output = attn @ outer
        output = output.sum(-1)
        return output


class FGN(nn.Module):

    def __init__(self, bert_model_name_or_path, num_embeddings, embedding_dim, k_c, s_c, k_g, s_g, d_c=512, d_g=64, dropout_prob=0.2):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name_or_path)
        self.cgs_cnn = CGSCNN(num_embeddings=num_embeddings, embedding_dim=embedding_dim, dropout_prob=dropout_prob)
        self.oos_sliding_window = OosSlidingWindow(k_c=k_c, s_c=s_c, k_g=k_g, s_g=s_g, d_c=d_c, d_g=d_g)
        self.slice_attention = SliceAttention(k_c=k_c, k_g=k_g)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_output = bert_output.last_hidden_state
        cgs_cnn_output = self.cgs_cnn(input_ids)