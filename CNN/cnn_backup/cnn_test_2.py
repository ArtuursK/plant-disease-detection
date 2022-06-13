

from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split

import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np



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

# model...
model = keras.models.Sequential()
model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding="valid", activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(256, 3, activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
print(model.summary())


# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]

model.compile(optimizer=optim, loss=loss, metrics=metrics)

# training
batch_size = 20
epochs = 5

modResult = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

# evaulate
model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)

#pd.DataFrame(modResult.history).plot(figsize=(15, 5))
#plt.show()

whereToSaveModel = '../SavedModels/CNN_Model_1'
if not os.path.exists(whereToSaveModel):
    os.makedirs(whereToSaveModel)

model.save("../SavedModels/CNN_Model_1")
print("Model was saved successfully")







# References:
# https://www.youtube.com/watch?v=eMMZpas-zX0
# https://www.youtube.com/watch?v=ad-Qc42Kbx8&ab_channel=DerekBanas
