# -*- encoding: utf-8 -*-
'''
@create_time: 2021/06/23 15:45:38
@author: lichunyu
'''

from typing import overload
import torch


# class A(object):

#     def __init__(cls, name, bases, attrs):
#         pass



def __init__(self, x):
    self.x = x
    print(self.x)


@classmethod
def cls_func(cls):
    pass


@staticmethod
def static_func():
    pass


# A = type(
#     'A',
#     (object,),
#     {
#         '__init__': __init__,
#         'cls_func': cls_func,
#         'static_func': static_func
#     }
# )
# a = A('x')

class A(object):

    def t(self):
        print('A')


class B(object):

    def t(self):
        print('B')


class C(A, B):
    ...



c = C()
c.t()