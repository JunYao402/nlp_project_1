# 用word2vec实现那几种清洗组合，通过ffn
from nlp_task2 import df
from vectorization import generate_sentence_vector
from model import generate_ffn_model
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y = df['label']
lower_vector = np.array(generate_sentence_vector(df['clean_text_lower']))
lower_remove_vector = np.array(generate_sentence_vector(df['clean_text_lower_remove']))
lower_remove_lemma_vector = np.array(generate_sentence_vector(df['clean_text_lower_remove_lemma']))

print("only lower")
generate_ffn_model(lower_vector,y)
print("lower and remove stop word")
generate_ffn_model(lower_remove_vector,y)
print("lower and remove stop word and lemma")
generate_ffn_model(lower_remove_lemma_vector,y)
