# -*- encoding: utf-8 -*-
'''
@create_time: 2021/06/23 14:34:36
@author: lichunyu
'''

from torch.autograd.functional import jacobian, hessian
from torch.nn import Linear, AvgPool2d
import torch


fc = Linear(4, 2)
pool = AvgPool2d(kernel_size=2)


def scalar_func(x):
    y = x ** 2
    z = torch.sum(y)
    return z


def vector_func(x):
    y = fc(x)
    return y


def mat_func(x):
    x = x.reshape((1,1,) + x.shape)
    x = pool(x)
    x = x.reshape(x.shape[2:])
    return x ** 2


vector_input = torch.randn(4, requires_grad=True)
mat_input = torch.randn((4, 4), requires_grad=True)

_ = mat_func(mat_input)

j = jacobian(scalar_func, vector_input)
assert j.shape == (4, )
assert torch.all(jacobian(scalar_func, vector_input) == 2 * vector_input)
h = hessian(scalar_func, vector_input)
assert h.shape == (4, 4)
assert torch.all(hessian(scalar_func, vector_input) == 2 * torch.eye(4))

j = jacobian(vector_func, vector_input)
assert j.shape == (2, 4)
assert torch.all(j == fc.weight)
# h = hessian(vector_func, vector_input)

j = jacobian(mat_func, mat_input)
assert j.shape == (2, 2, 4, 4)
pass