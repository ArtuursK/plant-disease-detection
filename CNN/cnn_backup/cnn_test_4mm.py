

from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense

import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


datadir = "../Healthy_n_Unhealthy_4_Training_CM"

#NOTE : Please enter Category Names same as folder name
categories=['HealthyLeaves', 'UnhealthyLeaves']

train_imgs = []
target_arr = []
for category in categories:
    print(f'loading category : {category}')
    path = os.path.join(datadir, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        train_imgs.append(img_array)
        target_arr.append(categories.index(category))
    print(f'loaded category:{category} successfully')

allImages = np.array(train_imgs)
target = np.array(target_arr)
print(allImages.shape) # (109, 256, 256, 3)

x_train, x_test, y_train, y_test = train_test_split(allImages, target, test_size=0.20, random_state=77, stratify=target)
print(f"x_train.shape {x_train.shape}")
print(f"x_test.shape {x_test.shape}")
print('Splitted Successfully')

# Normalize: 0,255 -> 0,1 an essential math trick for better performance
train_images, test_images = x_train / 255.0, x_test / 255.0

# kernel_size - how big is the filters (3x3 in this case)
# pooling layer
model = keras.models.Sequential()
model.add(Conv2D(filters=10, kernel_size=3, activation="relu", input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=2, padding="valid"))
model.add(Conv2D(filters=10, kernel_size=3, activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(filters=10, kernel_size=3, activation="relu"))
model.add(MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"])

batch_size = 64
epochs = 5
modResult = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, verbose=2)

# evaulate
model.evaluate(x_test, y_test, batch_size = batch_size, verbose = 2)


# OPTIONAL - Test a random sample image that the model has not seen:
# Read in image
plantImage = tf.io.read_file("../../HealthyLeaves/HealthyLeavesTrainPreprocessed/O_Healthy_20210208_102309.jpg")

# Turn file into a tensor
plantImage = tf.image.decode_image(plantImage)

# Resize image
plantImage = tf.image.resize(plantImage, size=[256, 256])

# Normalize data
plantImage = plantImage / 255

# Tells us how likely the image belongs to a class in our model
prediction = model.predict(tf.expand_dims(plantImage, axis=0))
print(f"tf.round(prediction): {tf.round(prediction)}")

predicted_class = categories[int(tf.round(prediction))]
print(f"The predicted_class is : {predicted_class} with prediction: {prediction}")








# References:
# https://www.youtube.com/watch?v=eMMZpas-zX0
# https://www.youtube.com/watch?v=ad-Qc42Kbx8&ab_channel=DerekBanas
