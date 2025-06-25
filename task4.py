from nlp_task2 import df
from vectorization import generate_sentence_vector
from a_preprocess import clean_pipeline
import numpy as np
import pandas as pd

# 随机抽取label是negative的15条text,返回series
df_negative = df[df['label']=='negative']['text'].sample(n=15)
# 清洗这15条text,返回的是list
df_negative_all_clean = clean_pipeline(df_negative,lower=True,remove_stop_word=True,lemma=True)
# print(df_negative_all_clean)
# print([s.split() for s in df_negative_all_clean])
# print(type(df_negative_all_clean))
# 得到这15条text的average of word vector,每个text有一个300维的word2vec的vector,是list
vector_result = generate_sentence_vector([s.split() for s in df_negative_all_clean])
print(vector_result[:1])
# 求相似度
def dot_product(vec1, vec2):
    return sum(a * b for a, b in zip(vec1, vec2))

def vector_norm(vec):
    return sum(a * a for a in vec) ** 0.5

def cosine_similarity(vec1, vec2):
    norm1 = vector_norm(vec1)
    norm2 = vector_norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0  # 避免除以0
    return dot_product(vec1, vec2) / (norm1 * norm2)

# 假设你已经有了一个 list，每个元素是一个句子的 vector（比如300维）：
vectors = vector_result  # 15个句子的vector，每个是list

n = len(vectors)
similarity_matrix = [[0 for _ in range(n)] for _ in range(n)]

for i in range(n):
    for j in range(n):
        if i == j:
            similarity_matrix[i][j] = 1.0  # 自己和自己
        elif i < j:
            sim = cosine_similarity(vectors[i], vectors[j])
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim  # 对称矩阵

# 打印结果
for row in similarity_matrix:
    print(row)
