# -*- encoding: utf-8 -*-
'''
@create_time: 2021/02/17 14:58:09
@author: lichunyu
'''

import numpy as np
from scipy.sparse.linalg import svds  # 返回的奇异值为升序，需要reserve
from scipy.linalg import svd

from math import log
import os
import sys

sys.path.append(os.getcwd())

from data.mini_corpus import list_docs, list_words, words2idx

X = np.zeros((len(list_words), len(list_docs)), dtype=float)

n_topic = min(len(list_words), len(list_docs)) - 1

def tf_idf(word:str, doc:str, list_docs:list):
    tf = doc.split(' ').count(word) / len(doc.split(' '))
    list_splited_docs = [doc.split(' ') for doc in list_docs]
    df_i = 0
    for i in list_splited_docs:
        if word in i:
            df_i += 1
    unlog_idf = len(list_docs) / df_i
    idf = log(unlog_idf)
    return tf * idf


for i, w in enumerate(list_words):
    for j, d in enumerate(list_docs):
        X[i, j] = tf_idf(w, d, list_docs)

u, sigma, v_t  = svds(X, k=n_topic)
# u, sigma, v_t  = svd(X)

word2vec_lsa = u

print(word2vec_lsa)
