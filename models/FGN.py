# -*- encoding: utf-8 -*-
'''
@create_time: 2021/08/16 14:10:21
@author: lichunyu
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class FGN(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass


class CGSCNN(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, dropout_prob=0.2):
        super().__init__()
        self.img_embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
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
        output = self.conv2d2(output)
        output = self.pool2d2(output)
        output = torch.reshape(output, (4, 64))
        output = self.max_pool1d(output)  # TODO 加 关于英文的处理
        return output


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
        c_s_group = None # TODO 关于c_s的slid window
        g_s_group = None # TODO 关于g_s的slid window
        c_s_group_r = c_s_group.repeat(1, self.n).reshape(self.n**2, -1).unsqueeze(dim=1)
        g_s_group_r = g_s_group.repeat(self.n, 1).unsqueeze(dim=1)
        outer = torch.einsum('bnc,bng->bncg', [c_s_group_r, g_s_group_r]).squeeze(dim=1).reshape(self.n, \
                    self.n, c_s.shape[-1], g_s.shape[-1]).flatten(start_dim=-2)
