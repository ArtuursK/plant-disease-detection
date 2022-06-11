
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import cv2
import time

datadir = "../Healthy_n_Unhealthy_4_Training_CM"

#NOTE : Please enter Category Names same as folder name
categories=['HealthyLeaves', 'UnhealthyLeaves']

flat_data_arr=[]
target_arr=[]
for category in categories:
    print(f'loading category : {category}')
    path=os.path.join(datadir, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        flat_data_arr.append(img_array.flatten())
        target_arr.append(categories.index(category))
    print(f'loaded category:{category} successfully')

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data)
df['Target'] = target
print(df)

#Splitting the data into training and testing data
x=df.iloc[:, :-1]
y=df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)
print('Splitted Successfully')

param_grid={'C':[0.1, 1, 10, 100], 'gamma':[0.0001, 0.001, 0.1, 1], 'kernel':['rbf', 'poly']} #https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
svc=svm.SVC(probability=True)
print("The training of the model is started, please wait for while as it may take few minutes to complete")
model = GridSearchCV(svc, param_grid)

start = time.time()
model.fit(x_train, y_train)
end = time.time()
print(f"Elapsed training time: {end - start} seconds")
print('The Model is trained with the given images')
print(model.best_params_)

y_pred = model.predict(x_test)
print("The predicted Data is :")
print(y_pred)

print("The actual data is:")
print(np.array(y_test))

# Now, evaluate the model with the test images by obtaining its classification report, confusion matrix, and accuracy score
print(f"The model is {accuracy_score(y_pred, y_test) * 100}% accurate")

print("Classification report:")
print(classification_report(y_pred, y_test))

print("Confusion matrix:")
print(confusion_matrix(y_pred, y_test))


whereToSaveModel = '../SavedModels/SVM_Model_1'
if not os.path.exists(whereToSaveModel):
    os.makedirs(whereToSaveModel)

pickle.dump(model, open(whereToSaveModel + "/model.p", "wb"))
print("Model was saved successfully")

# References:
#SVM implementation in python
#https://medium.com/analytics-vidhya/image-classification-using-machine-learning-support-vector-machine-svm-dc7a0ec92e01





