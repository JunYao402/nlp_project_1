import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont
from matplotlib import font_manager
from nltk.corpus import stopwords
from numpy.core.defchararray import strip
from collections import Counter
import string
import wordcloud
import nltk
from wordcloud import WordCloud
from a_preprocess import positive_count, negative_count, neutral_count, word_positive, word_negative, word_neutral
from a_preprocess import remove_stopword_pun

nltk.download('stopwords')
stop_word = set(stopwords.words('english'))

# -------------------------------------------生成图-------------------------------------
if __name__ == "__main__":
    plt.figure()
    #   直方图，表示，每种class都有多少条评论（已做完）
    # 1.生成直方图，x是三个类别的柱子，y是每个类别有多少条review
    x = np.array(["positive", "negative", "neutral"])
    y = np.array([positive_count, negative_count, neutral_count])
    # 三个柱子的数值
    plt.text(x[0], y[0] + 100, str(positive_count), ha='center', va='top')
    plt.text(x[1], y[1] + 100, str(negative_count), ha='center', va='top')
    plt.text(x[2], y[2] + 100, str(neutral_count), ha='center', va='top')
    plt.bar(x, y)
    plt.title('number of reviews per class')

    # 2.柱形图，表示，每个class的平均长度（已做完）
    # positive的总单词数，除以总text个数，得到平均每个review有多少个词
    # 生成柱形图

    plt.figure()

    positive_length_two = float('%.2f' % (len(word_positive) / positive_count))
    negative_length_two = float('%.2f' % (len(word_negative) / negative_count))
    neutral_length_two = float('%.2f' % (len(word_neutral) / neutral_count))

    x_1 = np.array(["pos_len", "neg_len", "neu_len"])
    y_1 = np.array([positive_length_two, negative_length_two, neutral_length_two])
    plt.text(x_1[0], y_1[0] + 0.3, str(positive_length_two))
    plt.text(x_1[1], y_1[1] + 0.3, str(negative_length_two))
    plt.text(x_1[2], y_1[2] + 0.3, str(neutral_length_two))
    plt.title("average len of review per class ")
    plt.bar(x_1, y_1)

    #   云图（未做完，字体格式不对，chatgpt说下载一个）
    # 得到了label是positive的text，的单个的词的形式，type是list
    print(Counter(remove_stopword_pun(word_positive)).most_common(10))
    positive_common = Counter(remove_stopword_pun(word_positive)).most_common(10)

    plt.figure()
    p_c_x = [item[0] for item in positive_common]
    p_c_y = [item[1] for item in positive_common]
    plt.title("10 most common word in positive review")
    plt.bar(p_c_x, p_c_y)

    plt.figure()
    negative_common = Counter(remove_stopword_pun(word_negative)).most_common(10)
    n_c_x = [item[0] for item in negative_common]
    n_c_y = [item[1] for item in negative_common]
    plt.title("10 most common word in negative review")
    plt.bar(n_c_x, n_c_y)

    plt.figure()
    neutral_common = Counter(remove_stopword_pun(word_neutral)).most_common(10)
    neu_c_x = [item[0] for item in neutral_common]
    neu_c_y = [item[1] for item in neutral_common]
    plt.title("10 most common word in neutral review")
    plt.bar(neu_c_x, neu_c_y)

    plt.show()
