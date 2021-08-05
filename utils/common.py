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