from nlp_task2 import df

from gensim.models import Word2Vec
import numpy as np


# ---------word2vec-----------
# 输入值是分词后的句子列表
# def generate_word2vec_mode(x):
#     # word2vec的vectorization的模型，结果是用word2vec的vector
#     # 输入的是词的list，得到的是，每一个词的vector
#     word2vec_model = Word2Vec(sentences=x, vector_size=300, sg=1, window=5, min_count=1, epochs=10)
#     # 想得到句子的vector，把句子中每一个词，和word2vec中convert的词作比较
#     # 一一找到句子中的词的vector，然后求平均值。就是句子的vector

# 输入的是df按照某一个清洗组合后的列，是分词的list，的list
def generate_sentence_vector(sentence_list):
    word2vec_model = Word2Vec(sentences=sentence_list, vector_size=300, sg=1, window=5, min_count=1, epochs=10)
    vector_result = []
    for sentence in sentence_list:
        word_vec = [word2vec_model.wv[w] for w in sentence if w in word2vec_model.wv]
        if len(word_vec)>0:
            vector_result.append(np.mean(word_vec,axis=0))
        if len(word_vec)==0:
            all_zero = np.zeros(300)
            vector_result.append(all_zero)
    print(word2vec_model.wv.index_to_key[:10])
    return vector_result

# print(generate_sentence_vector([["i like apple"],["i love you"]]))
# 得到list,里面是，每个句子的300d的vector