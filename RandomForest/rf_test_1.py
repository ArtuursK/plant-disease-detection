

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime

import pickle
import cv2
import time
import pandas as pd
import os
import numpy as np

datadir = "../Healthy_n_Unhealthy_4_Training_CM"

#NOTE : Category Names must be the same as folder name
categories = ['HealthyLeaves', 'UnhealthyLeaves']

flat_data_arr=[]
target_arr=[]
for category in categories:
    print(f'loading category : {category}')
    path = os.path.join(datadir, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        flat_data_arr.append(img_array.flatten())
        target_arr.append(categories.index(category))
    print(f'loaded category:{category} successfully')

# Convert image 2D array to 1D together with its class
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data)
df['Target'] = target
#print(df)

# Splitting the data into training and testing data
x=df.iloc[:, :-1]
#print(f"x: {x}")
y=df.iloc[:, -1] # classes
#print(f"y: {y}")
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y)
print('Splitted Successfully')
print(f"x_train.shape: {x_train.shape}")
print(f"x_test.shape: {x_test.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"y_test.shape: {y_test.shape}")

# Next step includes the normalization of images followed by their reshaping.
# Normalization is a common step of image pre-processing and is achieved by simply dividing x_train by 255.0 for the train dataset
# and x_test by 255.0 for the test dataset.
# This is essential to maintain the pixels of all the images within a uniform range.

# Normalize: 0,255 -> 0,1 an essential math trick for better performance
x_train = x_train/255.0
x_test = x_test/255.0

# Implementing a Random Forest Classifier
model = RandomForestClassifier()

start = time.time()
model.fit(x_train, y_train)
end = time.time()
trainingduration = end - start
print(f"Elapsed training time: {trainingduration} seconds")
y_pred = model.predict(x_test)
print("The predicted Data is :")
print(y_pred)

print("The actual data is:")
print(np.array(y_test))

# Now, evaluate the model with the test images by obtaining its classification report, confusion matrix, and accuracy score
final_accuracy = accuracy_score(y_pred, y_test) * 100
print(f"The model is {final_accuracy}% accurate")

print("Classification report:")
print(classification_report(y_pred, y_test))

### For saving experiment results ################
clasifReport = classification_report(y_pred, y_test, output_dict=True)
print(clasifReport)

print(f"overall accuracy: {clasifReport['accuracy'] * 100}")
overallAccuracy = clasifReport['accuracy'] * 100
print(f"sensitivity (jutīgums): {clasifReport['0']['precision']}")
sensitivity = clasifReport['0']['precision']
print(f"specificity (specifiskums): {clasifReport['1']['precision']}")
specificity = clasifReport['1']['precision']
print(f"precision (precīzumspēja): {clasifReport['0']['recall']}")
precision = clasifReport['0']['recall']
print(f"NPV (Negatīvo atklāšanas biežums): {clasifReport['1']['recall']}")
npv = clasifReport['1']['recall']

with open("../Experiments/RF_results.csv", "a") as expfile:
    expfile.write(f"{datetime.now()},{overallAccuracy},{sensitivity},{specificity},{precision},{npv},{trainingduration}\n")
##################################################

print("Confusion matrix:")
print(confusion_matrix(y_pred, y_test))

whereToSaveModel = '../SavedModels/RF_Model_1'
if not os.path.exists(whereToSaveModel):
    os.makedirs(whereToSaveModel)

pickle.dump(model, open(whereToSaveModel + "/model.p", "wb"))
print("Model was saved successfully")



# OPTIONAL - test with random image:
print("Random image test")
plantImage = cv2.imread("../HealthyLeaves/HealthyLeavesTestPreprocessed/O_Healthy_IMG_2071.JPG")
l = [plantImage.flatten()]
probability=model.predict_proba(l)
for ind, val in enumerate(categories):
    print(f'{val} probability = {probability[0][ind]*100}%')
print("The predicted image is : " + categories[model.predict(l)[0]])




############################################    TEST    ###################################################################
# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# print(f"x_train.shape: {x_train.shape}")
# print(f"x_test.shape: {x_test.shape}")
# print(f"y_train.shape: {y_train.shape}")
# print(f"y_test.shape: {y_test.shape}")
#
# # Normalization
# x_train = x_train/255.0
# x_test = x_test/255.0
#
# #sklearn expects i/p to be 2d array-model.fit(x_train,y_train)=>reshape to 2d array
# nsamples, nx, ny, nrgb = x_train.shape
# x_train2 = x_train.reshape((nsamples,nx*ny*nrgb))
# print(f"x_train2.shape: {x_train2.shape}")


###############################################################################################################












# References:

# https://www.askpython.com/python/built-in-methods/python-iloc-function

# https://www.analyticsvidhya.com/blog/2022/01/image-classification-using-machine-learning/



# Notes
#
# 1). X_train - This includes your all independent variables,these will be used to train the model, also as we have specified the test_size = 0.4, this means 60% of observations from your complete data will be used to train/fit the model and rest 40% will be used to test the model.
#
# 2). X_test - This is remaining 40% portion of the independent variables from the data which will not be used in the training phase and will be used to make predictions to test the accuracy of the model.
#
# 3). y_train - This is your dependent variable which needs to be predicted by this model, this includes category labels against your independent variables, we need to specify our dependent variable while training/fitting the model.
#
# 4). y_test - This data has category labels for your test data, these labels will be used to test the accuracy between actual and predicted categories.







