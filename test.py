# -*- encoding: utf-8 -*-
'''
@create_time: 2021/08/31 14:38:27
@author: lichunyu
'''

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


w_list = load_yangjie_rich_pretrain_word_list(
    '/root/pretrain-models/flat/yangjie_word_char_mix.txt'
)
with open()
pass


a = [[1,2,3,4]]

a = [[1], [2], ]





def a():
    yield 1


A = a()
next(A)