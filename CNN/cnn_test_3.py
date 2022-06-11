
import pandas as pd

import os
import cv2

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense

import tensorflow.keras as keras
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from sklearn.model_selection import train_test_split


data_dir = "../Healthy_n_Unhealthy_4_Training_CM"
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

# kernel_size - how big is the filters (3x3 in this case)
# pooling layer
model = keras.models.Sequential()
model.add(Conv2D(filters=10,
                 kernel_size=3,
                 activation="relu",
                 input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=2, padding="valid"))
model.add(Conv2D(filters=10,
                  kernel_size=3,
                  activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(filters=10,
                 kernel_size=3,
                 activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
print(model.summary())

model.compile(loss="binary_crossentropy",
                optimizer=keras.optimizers.Adam(),
                metrics=["accuracy"])

modelResult = model.fit(training_data_norm,
                      epochs=5,
                      steps_per_epoch=len(training_data_norm),
                      validation_data=validation_data_norm,
                      validation_steps=len(validation_data_norm))

#pd.DataFrame(modelResult.history).plot(figsize=(20, 10))
#plt.show()


whereToSaveModel = '../SavedModels/CNN_Model_3'
if not os.path.exists(whereToSaveModel):
    os.makedirs(whereToSaveModel)

model.save(whereToSaveModel)
print("Model was saved successfully")






# References:
# https://www.youtube.com/watch?v=ad-Qc42Kbx8&ab_channel=DerekBanas
#
#












