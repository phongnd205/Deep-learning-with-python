#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 10:44:29 2019

@author: phongnd205
"""

import tensorflow
from tensorflow import keras
from keras.models import load_model

model1 = load_model('cats_and_dogs_small_2.h5')
model1.summary()
