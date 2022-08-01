# -*- encoding: utf-8 -*-
'''
@create_time: 2022/07/29 16:17:13
@author: lichunyu
'''

from .._base.base_trainer import ClassificationTrainer


class ModelTrainer(ClassificationTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)