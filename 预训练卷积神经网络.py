# -*- coding: utf-8 -*-
# Date: 2021/05/02

import keras

from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150,150,3))

