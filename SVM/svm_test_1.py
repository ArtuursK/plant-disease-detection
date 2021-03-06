
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime
from sklearn import metrics

import pandas as pd
import os
import numpy as np
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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y)
print('Splitted Successfully')

#param_grid={'C':[0.1, 1, 10, 100], 'gamma':[0.0001, 0.001, 0.1, 1], 'kernel':['rbf', 'poly']} #https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
param_grid={'C':[0.1], 'gamma':[0.0001], 'kernel':['linear']}
svc=svm.SVC(probability=True)
print("The training of the model is started, please wait for while as it may take few minutes to complete")
model = GridSearchCV(svc, param_grid)

start = time.time()
model.fit(x_train, y_train)
end = time.time()
trainingduration = end - start
print(f"Elapsed training time: {trainingduration} seconds")
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

### For saving experiment results ################
clasifReport = classification_report(y_pred, y_test, output_dict=True)
print(clasifReport)

print(f"overall accuracy: {clasifReport['accuracy'] * 100}")
overallAccuracy = clasifReport['accuracy'] * 100
print(f"sensitivity (jut??gums): {clasifReport['0']['precision']}")
sensitivity = clasifReport['0']['precision']
print(f"specificity (specifiskums): {clasifReport['1']['precision']}")
specificity = clasifReport['1']['precision']
print(f"precision (prec??zumsp??ja): {clasifReport['0']['recall']}")
precision = clasifReport['0']['recall']
print(f"NPV (Negat??vo atkl????anas bie??ums): {clasifReport['1']['recall']}")
npv = clasifReport['1']['recall']

with open("../Experiments/SVM_results.csv", "a") as expfile:
    expfile.write(f"{datetime.now()},{overallAccuracy},{sensitivity},{specificity},{precision},{npv},{trainingduration}\n")
##################################################

print("Confusion matrix:")
print(confusion_matrix(y_pred, y_test))


whereToSaveModel = '../SavedModels/SVM_Model_1'
if not os.path.exists(whereToSaveModel):
    os.makedirs(whereToSaveModel)

pickle.dump(model, open(whereToSaveModel + "/model.p", "wb"))
print("Model was saved successfully")


printROC = True
if printROC:
    y_pred_proba = model.predict_proba(x_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    print(f"AUC: {str(auc)}")
    df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'auc': auc})
    pd.DataFrame(df).to_csv("SVMROC.csv", index=False)

# References:
#SVM implementation in python
#https://medium.com/analytics-vidhya/image-classification-using-machine-learning-support-vector-machine-svm-dc7a0ec92e01





