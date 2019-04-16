#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 22:42:52 2019

@author: phongnd205
"""

import numpy as np

x1 = np.array(12)
print(x1)
print(x1.ndim)

x2 = np.array([12, 4, 6, 10])
print(x2)
print(x2.ndim)

x3 = np.array([[12, 4, 5],
               [10, 3, 4],
               [1, 31, 3]])
print(x3)
print(x3.ndim)

x4 = np.array([[[12, 4, 5],
               [10, 3, 4],
               [1, 31, 3]],
                [[11, 3, 1],
               [12, 5, 6],
               [1, 44, 4]]])
print(x4)
print(x4.ndim)


