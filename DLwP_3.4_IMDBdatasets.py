#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 22:20:16 2019

@author: phongnd205
"""

# import necessary library
import tensorflow
from tensorflow import keras
from keras.datasets import imdb
import numpy as np

"""
load list of reviews training data and testing data
each review is list of words,
num_words - number of most frequent word in subtitle
labels 0s-nagative, 1s-positive
"""
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(train_data.shape)
print(train_labels.shape)

# encoding the integer sequences into a binary matrix
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = keras.models.Sequential()
model.add(keras.layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))



model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

"""
loss - binary_crossentropy, mse
number of node in hidden - 16, 32, 64, ...
2, 3, 4, ... hidden layers
activation could be relu, tanh
"""

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# plot the training and validation loss
import matplotlib.pyplot as plt
history_dict = history.history
loss_value = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(history.history['acc']) + 1)

plt.plot(epochs, loss_value, 'bo', label='training loss')
plt.plot(epochs, val_loss_values, 'b', label='validation loss')
plt.title('training and validating loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# plot the training and validation accuracy
plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('training and validating accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# predict with the test data
model.predict(x_test)


