import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont
from matplotlib import font_manager
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from numpy.core.defchararray import strip
from collections import Counter
import string
import wordcloud
import nltk
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_word = set(stopwords.words('english'))

# task1
texts = []
labels = []

with open(r'C:\Users\张俊瑶\Desktop\JAVA笔记\NLP\FinancialPhraseBank\FinancialPhraseBank-v1.0\Sentences_50Agree.txt', 'r',
          encoding='ISO-8859-1')as f:
    lines = f.readlines()
for line in lines:
    clean_line = line.strip()
    if "@positive" in line:
        texts.append(line.replace("@positive", "").strip())
        labels.append("positive")
    if "@negative" in line:
        texts.append(line.replace("@negative", "").strip())
        labels.append("negative")
    if "@neutral" in line:
        texts.append(line.replace("@neutral", "").strip())
        labels.append("neutral")
df = pd.DataFrame({
    'text': texts,
    'label': labels
})
df.to_csv('all_data.csv', index=False, encoding='utf-8-sig')
# print(df.head(5))
# 到这里，是把老师给的格式改成了csv，名字叫all_data.csv

# 读取csv
all_data = pd.read_csv('all_data.csv')
positive_count = df[df['label'] == 'positive'].shape[0]
negative_count = df[df['label'] == 'negative'].shape[0]
neutral_count = df[df['label'] == 'neutral'].shape[0]


# 一个方法，input是要进行remove stop word和punctuation的变量
# remove stop word 和 标点
def remove_stopword_pun(x):
    words = [w.strip(string.punctuation) for w in x]
    words = [w for w in words if w and w not in stop_word]
    return words


# 整个csv表中，取label是positive的行，形成的表
df_positive = df[df['label'] == "positive"]
# label是positive的行形成的表中的text内容
text_positive = df_positive['text']
df_negative = df[df['label'] == "negative"]
text_negative = df_negative['text']
df_neutral = df[df['label'] == "neutral"]
text_neutral = df_neutral['text']

# label是positive的行的text
# lower，然后split
word_positive = []
for text in text_positive:
    word_positive.extend((text.lower()).split())

word_negative = []
for text in text_negative:
    word_negative.extend((text.lower()).split())

word_neutral = []
for text in text_neutral:
    word_neutral.extend((text.lower()).split())


# --------------task2,3--------------
def clean_pipeline(y,lower=True,remove_stop_word=True,lemma=True):

    if lower:
        y = [s.lower() for s in y]

    y = [word_tokenize(sentence) for sentence in y]

    if remove_stop_word:
        y = [[w for w in sentence if w.isalpha() and w not in stop_word] for sentence in y]

    if lemma:
        y = [[lemmatizer.lemmatize(w, pos='v') for w in sentence] for sentence in y]

        # 变回整句字符串
    y = [' '.join(sentence) for sentence in y]

    return y


