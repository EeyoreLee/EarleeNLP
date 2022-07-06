# -*- encoding: utf-8 -*-
'''
@create_time: 2022/06/24 14:12:33
@author: lichunyu
'''


def snake2upper_camel(name:str):
    """snake-case name convert to upper-camel-case. like {bert_classification} -> {BertClassification}

    :param name: snake-case name
    :type name: str
    """    
    return name.replace('_', ' ').title().replace(' ', '')