
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

from sklearn.model_selection import train_test_split



healthyPlantPath = os.path.dirname(os.path.abspath(__file__)) + "/HealthyLeaves/HealthyLeavesTrain/"



# plt.figure(figsize=(12,12))
# for i in range(1,10):
#     plt.subplot(3,3,i) # 3 x 3 matrix
#     plt.tight_layout()
#     rand_img = imread(healthyPlantPath + random.choice(os.listdir(healthyPlantPath)))
#     plt.imshow(rand_img)
#     plt.xlabel(rand_img.shape[1], fontsize = 10)#width of image
#     plt.ylabel(rand_img.shape[0], fontsize = 10)#height of image
#
# plt.show()

#Converting Images to array
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, (256,256))
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


#dir = "../input/leaf-image-dataset/Plant_images"
dir = os.path.dirname(os.path.abspath(__file__)) + "/HealthyUnhealthyLeaves"
root_dir = listdir(dir)
image_list, label_list = [], []
all_labels = ['Healthy', 'Unhealthy']
binary_labels = [0, 1]
temp = -1

# Reading and converting each image to numpy array
for directory in root_dir:
    plant_image_list = listdir(f"{dir}/{directory}")
    temp += 1
    for files in plant_image_list:
        image_path = f"{dir}/{directory}/{files}"
        image_list.append(convert_image_to_array(image_path))
        label_list.append(binary_labels[temp])

# 1 - unhealthy
# 0 - healthy


# Visualize the number of classes count
label_counts = pd.DataFrame(label_list).value_counts()
print(label_counts.head())

# Check the shape of the first image in image_list.
print("First image shape: ", image_list[0].shape)

# Checking the total number of the images which is the length of the labels list.
label_list = np.array(label_list)
print("total number of images: ", label_list.shape)


x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state = 10)
# Now we will normalize the dataset of our images.
# As pixel values ranges from 0 to 255 we will divide each image pixel with 255 to normalize the dataset.
x_train = np.array(x_train, dtype=np.float16) / 225.0
x_test = np.array(x_test, dtype=np.float16) / 225.0
x_train = x_train.reshape( -1, 256, 256, 3)
x_test = x_test.reshape( -1, 256, 256, 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Next we will create a network architecture for the model.
# We have used different types of layers according to their features namely:
# Conv_2d (It is used to create a convolutional kernel that is convolved with the input layer to produce the output tensor),
# max_pooling2d (It is a downsampling technique which takes out the maximum value over the window defined by poolsize),
# flatten (It flattens the input and creates a 1D output),
# Dense (Dense layer produce the output as the dot product of input and kernel).

model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(256, 256, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(8, activation="relu"))
model.add(Dense(2, activation="softmax"))
print(model.summary())



model.compile(loss = 'categorical_crossentropy', optimizer = Adam(0.0001), metrics=['accuracy'])

#Next we will split the dataset into validation and training data.
# Splitting the training data set into training and validation data sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)

# Training the model
epochs = 4
history = model.fit(x_train, y_train, batch_size = 1, epochs = epochs,
                    validation_data = (x_val, y_val))


#Plot the training history
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], color='r')
plt.plot(history.history['val_accuracy'], color='b')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])

plt.show()

print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")



# References:
# https://www.kaggle.com/code/hamedetezadi/leaf-disease-prediction-using-cnn/notebook
#
#












