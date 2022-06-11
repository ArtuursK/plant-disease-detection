
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


data_dir = "../../Healthy_n_Unhealthy_4_Training_CM"
training_data = keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training",
    batch_size=32, image_size=(256, 256), seed=66)

validation_data = keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation",
    batch_size=32, image_size=(256, 256), seed=66)

class_names = training_data.class_names
print(f"class names: ${class_names}")

norm_layer = keras.layers.Rescaling(1/255.)
training_data_norm = training_data.map(lambda x, y: (norm_layer(x), y))
validation_data_norm = validation_data.map(lambda x, y: (norm_layer(x), y))
image_batch, labels_batch = next(iter(training_data_norm))

# kerel_size - how big is the filters (3x3 in this case)
# pooling layer
model_5 = keras.models.Sequential([
    Conv2D(filters=10,
           kernel_size=3,
           activation="relu",
           input_shape=(256, 256, 3)),
    MaxPooling2D(pool_size=2, padding="valid"),
    Conv2D(filters=10,
           kernel_size=3,
           activation="relu"),
    MaxPooling2D(),
    Conv2D(filters=10,
           kernel_size=3,
           activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(1, activation="sigmoid")
])
model_5.compile(loss="binary_crossentropy",
                optimizer=keras.optimizers.Adam(),
                metrics=["accuracy"])

history_5 = model_5.fit(training_data_norm,
                      epochs=5,
                      steps_per_epoch=len(training_data_norm),
                      validation_data=validation_data_norm,
                      validation_steps=len(validation_data_norm))

pd.DataFrame(history_5.history).plot(figsize=(20, 10))
#plt.show()





# OPTIONAL - Test a random sample image that the model has not seen:
# Read in image
#plantImage = tf.io.read_file("../HealthyLeaves/HealthyLeavesTestPreprocessed/O_Healthy_IMG_2062.JPG")
plantImage = tf.io.read_file("../../HealthyLeaves/HealthyLeavesTrainPreprocessed/O_Healthy_20210208_102309.jpg")

# Turn file into a tensor
plantImage = tf.image.decode_image(plantImage)

# Resize image
plantImage = tf.image.resize(plantImage, size=[256, 256])

# Normalize data
plantImage = plantImage / 255

# Tells us how likely the image belongs to a class in our model
prediction = model_5.predict(tf.expand_dims(plantImage, axis=0))
predicted_class = class_names[int(tf.round(prediction))]
print(f"The predicted_class is : {predicted_class} with prediction: {prediction}")


#Save & Load Model
# Save Model
print("Saving model")
model_5.save("../SavedModels/CNN_Model_5")
# Load trained model
print("Loading model")
saved_model_5 = tf.keras.models.load_model("../SavedModels/CNN_Model_5")
#saved_model_5.evaluate(validation_data_norm)
prediction = saved_model_5.predict(tf.expand_dims(plantImage, axis=0))
predicted_class = class_names[int(tf.round(prediction))]
print(f"The predicted_class is : {predicted_class} with prediction: {prediction}")


# References:
# https://www.youtube.com/watch?v=ad-Qc42Kbx8&ab_channel=DerekBanas
#
#












