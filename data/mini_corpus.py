# -*- encoding: utf-8 -*-
'''
@create_time: 2021/02/17 14:11:21
@author: lichunyu
'''


list_docs = [
    'how are you',
    'I am fine',
    'you are beautiful',
    'thanks a lot',
    'see you next day'
]

# print(list(set((' ').join(list_docs).split(' '))))

list_words = ['you', 'lot', 'am', 'beautiful', 'thanks', 'next', 'fine', 'are', 'day', 'how', 'a', 'I', 'see']

# print({w:idx for idx, w in enumerate(list_words)})

words2idx = {'you': 0, 'lot': 1, 'am': 2, 'beautiful': 3, 'thanks': 4, 'next': 5, 'fine': 6, 'are': 7, 'day': 8, 'how': 9, 'a': 10, 'I': 11, 'see': 12}

