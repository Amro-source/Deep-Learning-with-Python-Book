# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:45:46 2019

@author: Zikantika
"""

from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))