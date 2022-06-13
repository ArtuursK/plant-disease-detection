

from tensorflow import keras
from sklearn.model_selection import train_test_split
import os
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime

import time
import pandas as pd
import cv2
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

x_train, x_test, y_train, y_test = train_test_split(allImages, target, test_size=0.20, stratify=target)
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
model.add(Conv2D(filters=10, kernel_size=3, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
#model.add(Dense(1, activation='softmax'))

model.compile(loss="binary_crossentropy",
              optimizer=keras.optimizers.Adam(),
              metrics=["accuracy"])

batch_size = 20
epochs = 5

start = time.time()
modResult = model.fit(x_train, y_train, epochs = epochs, verbose=2)
end = time.time()
trainingduration = end - start
print(f"Elapsed training time: {trainingduration} seconds")
#pd.DataFrame(modResult.history).plot(figsize=(20, 10))
#plt.show()

# evaluation
# print("Model evaluation:")
# model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)

y_pred = model.predict(x_test, batch_size=batch_size)
y_pred = np.round(y_pred)
print("The predicted Data is :")
print(y_pred)

print("The actual data is:")
print(np.array(y_test))

print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")

print("Classification report:")
print(classification_report(y_pred, y_test))

### For saving experiment results ################
clasifReport = classification_report(y_pred, y_test, output_dict=True)
print(clasifReport)

print(f"overall accuracy: {clasifReport['accuracy'] * 100}")
overallAccuracy = clasifReport['accuracy'] * 100
print(f"sensitivity (jutīgums): {clasifReport['0.0']['precision']}")
sensitivity = clasifReport['0.0']['precision']
print(f"specificity (specifiskums): {clasifReport['1.0']['precision']}")
specificity = clasifReport['1.0']['precision']
print(f"precision (precīzumspēja): {clasifReport['0.0']['recall']}")
precision = clasifReport['0.0']['recall']
print(f"NPV (Negatīvo atklāšanas biežums): {clasifReport['1.0']['recall']}")
npv = clasifReport['1.0']['recall']

with open("../Experiments/CNN_results.csv", "a") as expfile:
    expfile.write(f"{datetime.now()},{overallAccuracy},{sensitivity},{specificity},{precision},{npv},{trainingduration}\n")
##################################################

print("Confusion matrix:")
print(confusion_matrix(y_pred, y_test))

whereToSaveModel = '../SavedModels/CNN_Model_3'
if not os.path.exists(whereToSaveModel):
    os.makedirs(whereToSaveModel)

model.save(whereToSaveModel)
print("Model was saved successfully")





# References:
# https://www.youtube.com/watch?v=eMMZpas-zX0
# https://www.youtube.com/watch?v=ad-Qc42Kbx8&ab_channel=DerekBanas
