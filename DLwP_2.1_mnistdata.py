#!/usr/bin/env python355
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 22:01:45 2019

@author: phongnd205
"""

import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist

# input data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images.shape
test_images.shape

# network structure
network = keras.models.Sequential()
network.add(keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(keras.layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

# preparing data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# preparing labels
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# training the models
network.fit(train_images, train_labels,
            epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)






