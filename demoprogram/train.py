#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 4/6/21 11:08 PM
#@Author: Yiyang Huo
#@File  : train.py.py



from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from PIL import Image
from random import shuffle, choice
import numpy as np
from keras.utils import to_categorical
import os

# This is the image set we use to put into the model
IMAGE_SIZE = 256

# These are the path to training and test data set
IMAGE_TRAIN_DIRECTORY = './data/training_set'
IMAGE_TEST_DIRECTORY = './data/test_set'


# This turns the name into label that can be read by the model
# for example, if the image is cat, this function will return label [1, 0]
def label_img(name):
    if name == 'cats': return np.array([1, 0])
    elif name == 'notcats' : return np.array([0, 1])


# This function load the image to np arrays, a kind of data that can be recognized by model
def load_data(directory):
    print("Loading images...")
    data = [] # this is the data we need to return
    directories = next(os.walk(directory))[1]# path routing, not important

    for dirname in directories:
        print("Loading {0}".format(dirname))
        file_names = next(os.walk(os.path.join(directory, dirname)))[2]# path routing, not important
        for i in range(600):# ramdomly choose 600 images below
            image_name = choice(file_names)
            image_path = os.path.join(directory, dirname, image_name)

            # get the label of the image
            label = label_img(dirname)
            if "DS_Store" not in image_path: # besides DS_Store, all files are images

                # This three lines convert the image to a 'L' encoded, 256*256 images
                img = Image.open(image_path)
                img = img.convert('L')
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)


                arrayimg = np.array(img) # convert image to a np array
                data.append([arrayimg, label]) # add array to the end of "data"

    return data # return data


# This function is basically the same as load_data, However, it does not load image data randomly, so it is suitable for
# training instead of testing
def load_training_data():
    train_data = []
    directories = next(os.walk(IMAGE_TRAIN_DIRECTORY))[1]
    print("Loading images...")
    for dirname in directories:
        print("Loading {0}".format(dirname))
        image_dir_path = os.path.join(IMAGE_TRAIN_DIRECTORY, dirname)
        for img in os.listdir(image_dir_path):

            label = label_img(dirname)
            path = os.path.join(image_dir_path, img)
            if "DS_Store" not in path:
                img = Image.open(path)
                img = img.convert('L')
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
                train_data.append([np.array(img), label])

    shuffle(train_data)
    return train_data


# This function set up the keras CNN model, same as what in Discussion, However, add multiple layers
def training_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    return model


# Main function, program enter from here
if __name__ == "__main__":
    training_data = load_training_data()#load training data
    model = training_model()#retrieve model

    # Two lines below adjust extract label and data from the training_data
    training_images = np.array([i[0] for i in training_data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    training_labels = np.array([i[1] for i in training_data])

    # Same as what we have learnt in Discussion
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(training_images, training_labels, batch_size=50, epochs=10, verbose=1)

    # Save trained model
    model.save("model.h5")