# -*- encoding: utf-8 -*-
'''
@create_time: 2021/07/29 16:28:26
@author: lichunyu
'''


from fastNLP import Vocabulary
from fastNLP import DataSet


dataset = DataSet({'chars': [
                                ['今', '天', '天', '气', '很', '好', '。'],
                                ['被', '这', '部', '电', '影', '浪', '费', '了', '两', '个', '小', '时', '。']
                            ],
                    'target': ['neutral', 'negative']
})

vocab = Vocabulary()
vocab.from_dataset(dataset, field_name='chars')
vocab.index_dataset(dataset, field_name='chars')

target_vocab = Vocabulary(padding=None, unknown=None)
target_vocab.from_dataset(dataset, field_name='target')
target_vocab.index_dataset(dataset, field_name='target')
print(dataset)
