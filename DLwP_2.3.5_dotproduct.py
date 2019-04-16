#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:48:42 2019

@author: phongnd205
"""

import numpy as np

def naive_matrix_vector_dot(x,y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    
    z = np.zeros(x.shape[0])
    for idx in range(x.shape[0]):
        for jdx in range(x.shape[1]):
            z[idx] = x[idx, jdx] * y[jdx]
    return z
        