#!/usr/bin/env python
# coding: utf-8

import keras 
import numpy as np


from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# 向量化函数
def vectorize_sequences(sequences, dimension=10000):
    # 25000*10000 的矩阵
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        # 每条评论的单词索引，对应位置设为1
        results[i, sequence] = 1. # 参考numpy的高级索引
    return results

# 训练数据和测试数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

test = vectorize_sequences(train_data[0:3])

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')



from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))


model.summary()


model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])



x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train,epochs=4, batch_size=512, validation_data=(x_val, y_val))

history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history.history['accuracy']  # 训练资料的正确性
val_acc = history.history['val_accuracy']  # val 测试资料的正确性
loss = history.history['loss']  # 训练资料的损失值
val_loss = history.history['val_loss']  # val 测试资料的损失值

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label="Training loss")
plt.plot(epochs, val_loss, 'b', label="Validation loss")
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()

plt.plot(epochs,acc , 'bo', label="Training accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation accuracy")
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

results = model.evaluate(x_test, y_test)
model.predict(x_test)




