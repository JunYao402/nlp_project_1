from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from nlp_task2 import df
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# ------------------ffn--------------
# x是text.y是label。要变成array的形式

#y应该是df['label']
def generate_ffn_model(x,y):
    # 把df的label的列，转换成整数型，用于测试
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # 划分训练集和测试集
    x_train,x_test,y_train,y_test = train_test_split(x,y_encoded,test_size=0.2,random_state=42)
    #x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    # 把y分成3类，并改写
    num_classes = len(set(y_train))
    y_train_vector = to_categorical(y_train,num_classes)
    #y_valid_vector = to_categorical(y_valid,num_classes)
    y_test_vector = to_categorical(y_test,num_classes)

    model = Sequential()
    model.add(Dense(128, input_shape=(x_train.shape[1],), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. 训练模型
    model.fit(
        x_train, y_train_vector,
        #validation_data=(x_valid, y_valid_vector),
        epochs=10,
        batch_size=32,
        verbose=0
    )

    # 5. 预测 + 输出结果
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot()
    plt.show()

    return model