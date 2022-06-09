
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import cv2

#datadir = "../Healthy_n_Unhealthy_4_Training_CM"
datadir = "../Healthy_n_Unhealthy_4_Training_OG"

categories=['HealthyLeaves', 'UnhealthyLeaves']

#test the model with individual images that model has not seen before:
model = pickle.load(open("../SavedModels/SVM_Model_3/model.p", "rb"))
#plantImage = cv2.imread("../UnhealthyLeaves/UnhealthyLeavesTestPreprocessed/O_Unhealthy_20210208_104451.jpg")
plantImage = cv2.imread("../HealthyLeaves/HealthyLeavesTestPreprocessed/O_Healthy_IMG_2071.JPG")


plt.imshow(plantImage)
plt.show()
l = [plantImage.flatten()]
probability=model.predict_proba(l)
for ind, val in enumerate(categories):
    print(f'{val} probability = {probability[0][ind]*100}%')
print("The predicted image is : " + categories[model.predict(l)[0]])

# OPTIONAL - retrain model manually
# flat_data_arr=[]
# target_arr=[]
# for category in categories:
#     print(f'loading category : {category}')
#     path=os.path.join(datadir, category)
#     for img in os.listdir(path):
#         img_array=cv2.imread(os.path.join(path, img))
#         flat_data_arr.append(img_array.flatten())
#         target_arr.append(categories.index(category))
#     print(f'loaded category:{category} successfully')
# param_grid={'C':[0.1, 1, 10, 100], 'gamma':[0.0001, 0.001, 0.1, 1], 'kernel':['rbf', 'poly']}
# svc=svm.SVC(probability=True)
# print(f'Is the image a {categories[model.predict(l)[0]]} ?(y/n)')
# while(True):
#     b=input()
#     if(b == "y" or b =="n"):
#         break
#     print("please enter either y or n")
#
#     if(b =='n'):
#         print("What is the image?")
#         for i in range(len(categories)):
#             print(f"Enter {i} for {categories[i]}")
#         k=int(input())
#         while(k < 0 or k >= len(categories)):
#             print(f"Please enter a valid number between 0-{len(categories)-1}")
#             k=int(input())
#         print("Please wait for a while for the model to learn from this image :)")
#         flat_arr=flat_data_arr.copy()
#         tar_arr=target_arr.copy()
#         tar_arr.append(k)
#         flat_arr.extend(l)
#         tar_arr = np.array(tar_arr)
#         flat_df = np.array(flat_arr)
#         df1 = pd.DataFrame(flat_df)
#         df1['Target'] = tar_arr
#         model1 = GridSearchCV(svc, param_grid)
#         x1=df1.iloc[:, :-1]
#         y1=df1.iloc[:, -1]
#         x_train1, x_test1, y_train1, y_test1=train_test_split(x1, y1, test_size=0.20, random_state=77, stratify=y1)
#         d={}
#         for i in model.best_params_:
#             d[i]=[model.best_params_[i]]
#         model1 = GridSearchCV(svc, d)
#         model1.fit(x_train1, y_train1)







# References:

#SVM implementation in python
#https://medium.com/analytics-vidhya/image-classification-using-machine-learning-support-vector-machine-svm-dc7a0ec92e01





