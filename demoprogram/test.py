#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 4/7/21 12:13 AM
#@Author: Yiyang Huo
#@File  : test.py

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from PIL import Image
from random import shuffle, choice
import numpy as np
from keras.utils import to_categorical
import os
from train import load_data

IMAGE_SIZE = 256
IMAGE_TRAIN_DIRECTORY = './data/training_set'
IMAGE_TEST_DIRECTORY = './data/test_set'


if __name__ == "__main__":
    # still, load data
    test_data = load_data('./data/test_set')

    # Modify the data, just like train.py
    test_images = np.array([i[0] for i in test_data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    test_labels = np.array([i[1] for i in test_data])

    # load the model
    model = load_model("model.h5")

    # use the model and test data to evaluate accuracy
    loss, acc = model.evaluate(test_images, test_labels, verbose=1)
    print("accuracy: {0}".format(acc * 100))