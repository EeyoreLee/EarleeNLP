# -*- encoding: utf-8 -*-
'''
@create_time: 2021/07/01 16:14:07
@author: lichunyu
'''

import torch


A = torch.tensor([[1,2,3],[4,5,6]])
B = torch.tensor([[2,2,1],[3,2,1]])

C = torch.mm(A, B.T)
D = torch.einsum('ab,cb->ac', A, B)
E = torch.einsum('ij->ji', A)
pass