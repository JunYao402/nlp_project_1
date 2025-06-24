import nltk
from nltk.corpus import stopwords
from a_preprocess import df
from a_preprocess import clean_pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
lemmatizer = nltk.WordNetLemmatizer()
stop_word = set(stopwords.words('english'))

test_bayes_1 = df['text']
df['clean_text_lower'] = clean_pipeline(test_bayes_1, lower=True, remove_stop_word=False, lemma=False)
df['clean_text_lower_remove'] = clean_pipeline(test_bayes_1, lower=True, remove_stop_word=True, lemma=False)
df['clean_text_lower_remove_lemma'] = clean_pipeline(test_bayes_1, lower=True, remove_stop_word=True, lemma=True)


def compare(input):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['label'])  # 原来是文字，现在是整数
    from sklearn.model_selection import train_test_split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        #    df['clean_text_lower_remove_lemma'],  # 换成你选用的那一列
        input,
        y_encoded,
        test_size=0.2,
        random_state=42
    )
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.utils import to_categorical

    # 如果是多分类，要做 one-hot 编码
    num_classes = len(set(y_encoded))
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # 模型结构
    model = Sequential()
    model.add(Dense(128, input_shape=(X_train_vec.shape[1],), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # 多分类用 softmax

    # 编译
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练
    model.fit(X_train_vec.toarray(), y_train_cat,  # 稀疏矩阵转 array
              validation_data=(X_test_vec.toarray(), y_test_cat),
              epochs=20, batch_size=32)
    from sklearn.metrics import classification_report
    import numpy as np

    y_pred_prob = model.predict(X_test_vec.toarray())
    y_pred = np.argmax(y_pred_prob, axis=1)
    print(classification_report(y_test, y_pred, target_names=le.classes_))


print("only lower")
compare(df['clean_text_lower'])
print("lower+remove")
compare(df['clean_text_lower_remove'])
print("lower+remove+lemma")
compare(df['clean_text_lower_remove_lemma'])
