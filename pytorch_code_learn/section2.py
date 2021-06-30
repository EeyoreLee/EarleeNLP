# -*- encoding: utf-8 -*-
'''
@create_time: 2021/06/23 11:42:11
@author: lichunyu
'''

import torch


def autograd(grad_fn, gradient):
    auto_grad = {}
    queue = [[grad_fn, gradient]]
    while queue != []:
        item = queue.pop()
        gradients = item[0](item[1])
        functions = [x[0] for x in item[0].next_functions]
        if type(gradients) is not tuple:
            gradients = (gradients,)
        for grad, func in zip(gradients, functions):
            if type(func).__name__ == 'AccumulateGrad':
                if hasattr(func.variable, 'auto_grad'):
                    func.variable.auto_grad = func.variable.auto_grad + grad
                else:
                    func.variable.auto_grad = grad
            else:
                queue.append([func, grad])


A = torch.tensor([3.], requires_grad=True)
B = torch.tensor([2.], requires_grad=True)
C = A ** 2
D = B ** 2
E = C * D
F = D + E


autograd(F.grad_fn, torch.tensor(1))
print(A.auto_grad, B.auto_grad)
pass