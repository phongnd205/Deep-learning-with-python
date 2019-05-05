#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 22:05:45 2019

@author: phongnd205
"""
#import tensorflow
from tensorflow import keras
from keras.datasets import boston_housing


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(64, keras.regularizers.kernel_regularizers.l2(0.001),
                                 activation='relu',
                                 input_shape=(train_data.shape[1],)))
    model.add(keras.layers.Dense(64, keras.regularizers.kernel_regularizers.l2(0.001),
                                 activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


import numpy as np

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_score = []

for idx in range(k):
    print('processing fold #%d' % idx)
    val_data = train_data[idx * num_val_samples: (idx + 1) * num_val_samples]
    val_targets = train_targets[idx * num_val_samples: (idx + 1) * num_val_samples]
    
    partial_train_data = np.concatenate(
            [train_data[:idx * num_val_samples],
             train_data[(idx + 1)*num_val_samples:]],
            axis=0)
    partial_train_targets = np.concatenate(
            [train_targets[:idx * num_val_samples],
             train_targets[(idx + 1)*num_val_samples:]],
            axis=0)
    
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_score.append(val_mae)
    
# saving the validation logs at each fold
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]
    
    
    partial_train_data = np.concatenate(
            [train_data[:idx * num_val_samples],
             train_data[(idx + 1)*num_val_samples:]],
            axis=0)
    partial_train_targets = np.concatenate(
            [train_targets[:idx * num_val_samples],
             train_targets[(idx + 1)*num_val_samples:]],
            axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        epochs=num_epochs, batch_size=1, verbose=0)
    
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    
    
average_mae_history = [
        np.mean(x[i] for x in all_mae_histories) for i in range(num_epochs)]
    
import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

