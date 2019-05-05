#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 21:44:46 2019

@author: phongnd205
"""

import tensorflow
from tensorflow import keras

conv_base = keras.applications.VGG16(weights='imagenet',
                                     include_top=False,
                                     input_shape=(150, 150, 3))
conv_base.summary()