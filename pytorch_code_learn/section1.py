# -*- encoding: utf-8 -*-
'''
@create_time: 2021/06/23 09:36:18
@author: lichunyu
'''

import torch
from torch._C import ParameterDict
from torch.autograd.function import Function

from typing import Any


class GradCoeff(Function):

    @staticmethod
    def forward(ctx: Any, x, coeff) -> Any:
        result = x ** 0.5 * coeff
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        result, = ctx.saved_tensors
        return grad_outputs * result, grad_outputs * result * 2.



a = torch.tensor([1.2], requires_grad=True)
b = torch.tensor([.5], requires_grad=True)
func = GradCoeff.apply(a, b)
func.backward()



pass