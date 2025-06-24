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

# 用不同组合清洗的列
test_bayes_1 = df['text']
df['clean_text_lower'] = clean_pipeline(test_bayes_1,lower=True,remove_stop_word=False,lemma=False)
df['clean_text_lower_remove'] = clean_pipeline(test_bayes_1,lower=True,remove_stop_word=True,lemma=False)
df['clean_text_lower_remove_lemma'] = clean_pipeline(test_bayes_1,lower=True,remove_stop_word=True,lemma=True)

if __name__ == "__main__":
    # 假设你有一个 DataFrame，包含 'cleaned_text' 和 'label' 两列
    # X 是清洗过的文本，y 是标签
    X1 = df['clean_text_lower']      # 你自己清洗好的列名
    y1 = df['label']                  # 标签列

    # 1. 划分训练集和测试集（80%/20%）
    X_train_text, X_test_text, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

    # 2. TF-IDF 向量化
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    # 3. 初始化 & 训练 Naive Bayes 模型
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # 4. 预测
    y_pred = model.predict(X_test_vec)

    # 5. 输出分类报告
    print("only lower")
    print(classification_report(y_test, y_pred))

    X2 = df['clean_text_lower_remove']      # 你自己清洗好的列名
    y2 = df['label']                  # 标签列

    # 1. 划分训练集和测试集（80%/20%）
    X_train_text, X_test_text, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

    # 2. TF-IDF 向量化
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    # 3. 初始化 & 训练 Naive Bayes 模型
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # 4. 预测
    y_pred = model.predict(X_test_vec)

    # 5. 输出分类报告
    print("lower and remove stop")
    print(classification_report(y_test, y_pred))

    X3 = df['clean_text_lower_remove_lemma']      # 你自己清洗好的列名
    y3 = df['label']                  # 标签列

    # 1. 划分训练集和测试集（80%/20%）
    X_train_text, X_test_text, y_train, y_test = train_test_split(X3, y3, test_size=0.2, random_state=42)

    # 2. TF-IDF 向量化
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    # 3. 初始化 & 训练 Naive Bayes 模型
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # 4. 预测
    y_pred = model.predict(X_test_vec)

    # 5. 输出分类报告
    print("all")
    print(classification_report(y_test, y_pred))




