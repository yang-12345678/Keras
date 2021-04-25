# -*- coding: utf-8 -*-
# Date: 2021/04/19

# 互联网电影数据库获取数据集
from keras.datasets import imdb
import numpy as np

(train_data, train_label), (test_data, test_label) = imdb.load_data(num_words=10000)

# word_index是一个将单词映射成整数索引的字典
word_index = imdb.get_word_index()

# 键值颠倒，整数索引映射为单词
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ''.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# 整数序列编码为二进制矩阵
def vectoirze_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    # enummerate枚举出索引值和对应的值
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results


# 将训练数据和测试数据向量化
x_train = vectoirze_sequences(train_data)
x_test = vectoirze_sequences(test_data)
# 标签向量化
y_train = np.asarray(train_label).astype('float32')
y_test = np.asarray(test_label).astype('float32')

# 架构选择：两个中间层，每层都有16个隐藏单元，使用relu作为激活函数，。
# 第三层输出一个标量，预测当前评论的情感。
# 最后一层使用sigmod激活输出概率值

# 完成对模型的定义
from keras import models
from keras import layers

model = models.Sequential()  # 按顺序
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))  # Dense表示一个全连接层
model.add(layers.Dense(16, activation='tanh'))
model.add(layers.Dense(1, activation='sigmoid'))

# 配置优化器
from keras import optimizers
from keras import losses
from keras import metrics

# 模型编译(选择优化器，选择损失函数)
# model.compile(optimizer='resprop',loss='binary_crossentropy',metrics=['accuracy'])
# 自定义优化器，损失和指标
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

# 在训练集中流出样本作为验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
# 训练模型
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

import matplotlib.pyplot as plt

history_dict = history.history
# 验证损失和训练损失
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
# 设置x轴数据，y轴数据，曲线格式，设置图例名称
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# 设置标题
plt.title('Training and Validation loss')
# 设置横纵坐标名称
plt.xlabel('Epochs')
plt.ylabel('loss')
# 显示图例
plt.legend()
plt.show()

acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 重新训练模型
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=512)

# 修改处：1修改隐藏层层数，2修改隐藏单元，3使用mse替代binary_crossentropy, 4使用tanh激活函数替代relu
# 原始数据预处理，化为张量转换到神经网络中
# 二分类问题的sigmod标量输出，使用binary_crossentropy损失函数。
# rmsprop优化器通常都是不错的选择
# 过拟合会导致数据效果越来越差
results = model.evaluate(x_test, y_test)
