
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.image import imread

import os, random
from os import listdir

import cv2

from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, array_to_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dropout, Dense

import tensorflow.keras as keras
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from sklearn.model_selection import train_test_split


healthyPlantPath = os.path.dirname(os.path.abspath(__file__)) + "/HealthyLeaves/HealthyLeavesTrain/"

data_dir = "../HealthyUnhealthyLeavesTrain"
training_data = keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training",
    batch_size=32, image_size=(256,256), seed=66)

validation_data = keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation",
    batch_size=32, image_size=(256,256), seed=66)

class_names = training_data.class_names
print(class_names)

norm_layer = keras.layers.Rescaling(1/255.)
training_data_norm = training_data.map(lambda x, y: (norm_layer(x), y))
validation_data_norm = validation_data.map(lambda x, y: (norm_layer(x), y))
image_batch, labels_batch = next(iter(training_data_norm))
print(image_batch[0])

tf.random.set_seed(66)
#initial
# model_1 = Sequential([
#     Flatten(input_shape=(256, 256, 3)),
#     Dense(4, activation="relu"),
#     Dense(4, activation="relu"),
#     Dense(1, activation="sigmoid")
# ])

model_1 = Sequential([
    Flatten(input_shape=(256, 256, 3)),
    Dense(100, activation="relu"),
    Dense(100, activation="relu"),
    Dense(100, activation="relu"),
    Dense(1, activation="sigmoid")
])

model_1.compile(loss="binary_crossentropy",
                optimizer=keras.optimizers.Adam(),
                metrics=["accuracy"])

history_1 = model_1.fit(training_data_norm, epochs=5,
                        validation_data=validation_data_norm,
                        validation_steps=len(validation_data_norm))


print(model_1.summary())





# References:
# https://www.youtube.com/watch?v=ad-Qc42Kbx8&ab_channel=DerekBanas
#
#












