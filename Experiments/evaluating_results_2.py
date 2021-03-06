

from numpy import mean

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

randomForestResults = "backup/RF_results.csv"
cnnResults = "backup/CNN_results.csv"
svmResults = "SVM_results.csv"

rfData = pd.read_csv(randomForestResults)
cnnData = pd.read_csv(cnnResults)
svmData = pd.read_csv(svmResults)

# AVG training duration
# metric = 'trainingduration'
# avg_overall_accuracy = {
#     'RandomForest': rfData[metric],
#     'CNN': cnnData[metric],
#     'SVM': svmData[metric]
# }
#
# fig, ax = plt.subplots()
# ax.boxplot(avg_overall_accuracy.values(), showmeans=True, meanline=True)
# ax.set_xticklabels(avg_overall_accuracy.keys())
# plt.ylabel("Apmācības laiks (sekundes)")
# plt.grid(axis='y')
# plt.show()

# AVG training duration
# height = [mean(rfData['trainingduration']), mean(cnnData['trainingduration']), mean(svmData['trainingduration'])]
# bars = ('RandomForest', 'CNN', 'SVM')
# plt.barh(bars, height)
# for index, value in enumerate(height):
#     plt.text(value, index, str(round(value, 2)))
# plt.xlabel('Apmācības laiks (sekundes)')
# plt.grid(axis='x')
# plt.show()


# AVG overall accuracy
# height = [mean(rfData['overallAccuracy']), mean(cnnData['overallAccuracy']), mean(svmData['overallAccuracy'])]
# bars = ('RandomForest', 'CNN', 'SVM')
# plt.barh(bars, height)
# for index, value in enumerate(height):
#     plt.text(value, index, str(round(value,2)))
# plt.xlabel('Kopējā precizitāte %')
# plt.xlim([0, 100])
# plt.grid(axis='x')
# plt.show()


# AVG overall accuracy boxplot
# avg_overall_accuracy = {
#     'RandomForest': rfData['overallAccuracy'],
#     'CNN': cnnData['overallAccuracy'],
#     'SVM': svmData['overallAccuracy']
# }
#
# fig, ax = plt.subplots()
# ax.boxplot(avg_overall_accuracy.values(), showmeans=True, meanline=True)
# ax.set_xticklabels(avg_overall_accuracy.keys())
# plt.ylabel("Kopējā precizitāte %")
# plt.grid(axis='y')
# plt.show()


# AVG sensitivity duration
# metric = 'sensitivity'
# height = [mean(rfData[metric]), mean(cnnData[metric]), mean(svmData[metric])]
# bars = ('RandomForest', 'CNN', 'SVM')
# plt.barh(bars, height)
# for index, value in enumerate(height):
#     plt.text(value, index, str(round(value, 2)))
# plt.xlabel('Jutīgums')
# plt.grid(axis='x')
# plt.show()

# metric = 'specificity'
# height = [mean(rfData[metric]), mean(cnnData[metric]), mean(svmData[metric])]
# bars = ('RandomForest', 'CNN', 'SVM')
# plt.barh(bars, height)
# for index, value in enumerate(height):
#     plt.text(value, index, str(round(value, 2)))
# plt.xlabel('Specifiskums')
# plt.grid(axis='x')
# plt.show()


# metric = 'precision'
# height = [mean(rfData[metric]), mean(cnnData[metric]), mean(svmData[metric])]
# bars = ('RandomForest', 'CNN', 'SVM')
# plt.barh(bars, height)
# for index, value in enumerate(height):
#     plt.text(value, index, str(round(value, 2)))
# plt.xlabel('Precīzumspēja')
# plt.grid(axis='x')
# plt.show()



# metric = 'npv'
# height = [mean(rfData[metric]), mean(cnnData[metric]), mean(svmData[metric])]
# bars = ('RandomForest', 'CNN', 'SVM')
# plt.barh(bars, height)
# for index, value in enumerate(height):
#     plt.text(value, index, str(round(value, 2)))
# plt.xlabel('Negatīvo atklāšanas biežums')
# plt.grid(axis='x')
# plt.show()







# ROC curve (receiver operating characteristic curve) is a graph
# showing the performance of a classification model at all classification thresholds.
# This curve plots two parameters: True Positive Rate. False Positive Rate.

# create ROC curves
rfROC = pd.read_csv("../RandomForest/RFROC.csv")
cnnROC = pd.read_csv("../CNN/CNNROC.csv")
svmROC = pd.read_csv("../SVM/SVMROC.csv")
plt.plot(rfROC['fpr'], rfROC['tpr'], 'r', label="RandomForest, AUC="+str(round(rfROC['auc'][0], 2)))
plt.plot(cnnROC['fpr'], cnnROC['tpr'], 'g', label="CNN, AUC="+str(round(cnnROC['auc'][0], 2)))
plt.plot(svmROC['fpr'], svmROC['tpr'], 'b', label="SVM, AUC="+str(round(svmROC['auc'][0], 2)))

diagonalX = [0, 1]; diagonalY = [0, 1]
plt.plot(diagonalX, diagonalY, color='gray', linestyle='dotted')
plt.ylabel('Patieso pozitīvo koeficients (True positive rate)')
plt.xlabel('Nepatiesi pozitīvo koeficients (False positive rate)')
plt.legend()
plt.show()















# References:
# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
# https://www.statology.org/plot-roc-curve-python/

# https://www.youtube.com/watch?v=4jRBRDbJemM&ab_channel=StatQuestwithJoshStarmer


# https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/







