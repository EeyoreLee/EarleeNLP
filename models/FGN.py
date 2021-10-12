# -*- encoding: utf-8 -*-
'''
@create_time: 2021/08/16 14:10:21
@author: lichunyu
'''
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer
from PIL import Image
import numpy as np


class CGSCNN(nn.Module):

    def __init__(self, weights, dropout_prob=0.2, num_embeddings=50):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.img_embedding = nn.Embedding.from_pretrained(weights)
        self.conv3d = nn.Conv3d(1, 4, (3,3,3), 1, 1)
        self.conv3d2 = nn.Conv3d(4, 8, (3,3,3), 1, 1)
        self.conv2d0 = nn.Conv2d(8, 8, (3,3), (1,1))
        self.conv2d = nn.Conv2d(8, 16, (2,2), (1,1))
        self.pool2d = nn.MaxPool2d(2, 2)
        self.conv2d2 = nn.Conv2d(16, 32, (2,2), (1,1))
        self.conv2d3 = nn.Conv2d(32, 64, (2,2), (1,1))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, char_input):# TODO 加 关于英文的处理
        embed = self.img_embedding(char_input)
        batch_size, seq_len, _ = embed.shape
        embed = embed.view(batch_size, seq_len, 1, self.num_embeddings, self.num_embeddings)
        embed = embed.transpose(1,2)
        droped_embed = self.dropout(embed)
        output = self.conv3d(droped_embed)
        output = self.conv3d2(output)
        output = output.transpose(1,2)
        output = output.view(batch_size*seq_len, -1, self.num_embeddings, self.num_embeddings)
        output = self.conv2d0(output)
        output = self.pool2d(output)
        output = self.conv2d(output)
        output = self.pool2d(output)
        output = self.conv2d2(output)
        output = self.pool2d(output)
        output = self.conv2d3(output)
        output = self.pool2d(output)
        output = torch.reshape(output, (batch_size*seq_len, 4, 64))
        output = torch.max(output, dim=-2)[0]
        output = output.reshape(batch_size, seq_len, 64)
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

    def __init__(self, k_c, s_c, k_g, s_g, d_c=768, d_g=64):
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

    def __init__(self, n):
        super().__init__()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.slice_linear = nn.Linear(n, n)
        self.query = nn.Linear(n, n)

    def forward(self, outer):
        k = self.slice_linear(outer)
        k = self.sigmoid(k)
        k = k.transpose(-1, -2)
        q = self.query(outer)
        q = self.sigmoid(q)
        attn = self.softmax(q*k) # TODO bugfix
        output = attn @ outer
        output = output.sum(-1)
        return output


class CGS_Tokenzier(object):

    def __init__(self, idx_map) -> None:
        super().__init__()
        self.idx_map = idx_map

    def __call__(self, text, return_tensor='pt'):
        super.__call__()
        input_ids = []
        for i in text:
            if i in self.idx_map:
                input_ids.append(self.idx_map[i])
            else:
                input_ids.append(-100)

        if return_tensor == '':
            return [input_ids]
        elif return_tensor == 'pt':
            return torch.from_numpy(np.array([input_ids]))
        else:
            raise Exception('unsupport')

    @classmethod
    def from_pretained(cls, config_path):
        if os.path.isdir(config_path):
            vocab_path = os.path.join(config_path, 'ccfr_vocab.txt')
        elif os.path.isfile(config_path):
            vocab_path = config_path
        else:
            raise Exception('no file named ccfr_vocab.txt')

        with open(vocab_path, 'r') as f:
            char_list = f.read().splitlines()

        idx_map = {char: idx for idx, char in enumerate(char_list)}
        return cls(idx_map)


class FGN(nn.Module):

    def __init__(self, bert_model_name_or_path, cgs_cnn_weights, k_c=96, s_c=12, k_g=8, s_g=1, d_c=768, d_g=64, dropout_prob=0.2):
        super().__init__()
        self.bert_config = BertConfig.from_pretrained(bert_model_name_or_path)
        self.bert = BertModel(self.bert_config)
        self.cgs_cnn = CGSCNN(weights=cgs_cnn_weights, dropout_prob=dropout_prob)
        self.oos_sliding_window = OosSlidingWindow(k_c=k_c, s_c=s_c, k_g=k_g, s_g=s_g, d_c=d_c, d_g=d_g)
        self.n = int(((d_c - k_c) / s_c) + 1) * k_c * k_g
        self.slice_attention = SliceAttention(self.n)
        # self.lstm = nn.LSTM()

    def forward(self, input_ids, char_input_ids, attention_mask=None, token_type_ids=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_output = bert_output.last_hidden_state[:,1:-1,:]
        cgs_cnn_output = self.cgs_cnn(char_input_ids)
        featrue_fusion = self.oos_sliding_window(bert_output, cgs_cnn_output)
        feature = self.slice_attention(featrue_fusion)

        pass





if __name__ == '__main__':
    bert_model_path = '/ai/223/person/lichunyu/pretrain-models/bert-base-chinese'
    weights = torch.load('fgn_weights_gray.pth')
    fgn = FGN(bert_model_path, weights)

    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    batch_dict = bert_tokenizer('今天是晴天', return_tensors='pt')
    input_ids = batch_dict['input_ids']
    attention_mask = batch_dict['attention_mask']

    cgs_tokenizer = CGS_Tokenzier.from_pretained('/root/EarleeNLP')
    char_input_ids = cgs_tokenizer('今天是晴天')

    res = fgn(input_ids, char_input_ids, attention_mask=attention_mask)
    pass
