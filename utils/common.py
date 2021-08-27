# -*- encoding: utf-8 -*-
'''
@create_time: 2021/08/05 14:28:18
@author: lichunyu
'''


def print_info(*inp,islog=True,sep=' '):
    from fastNLP import logger
    if islog:
        print(*inp,sep=sep)
    else:
        inp = sep.join(map(str,inp))
        logger.info(inp)


def text_rm_space(text:str):
    offsets, pointer = [], 0
    for idx, char in enumerate(text):
        offsets.append((pointer, idx))
        if char != ' ':
            pointer += 1
    return text.replace(' ', ''), offsets




if __name__ == '__main__':
    print(text_rm_space('as fas ,fsfsf asdf'))