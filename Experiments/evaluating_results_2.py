

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
metric = 'trainingduration'
avg_overall_accuracy = {
    'RandomForest': rfData[metric],
    'CNN': cnnData[metric],
    'SVM': svmData[metric]
}

fig, ax = plt.subplots()
ax.boxplot(avg_overall_accuracy.values(), showmeans=True, meanline=True)
ax.set_xticklabels(avg_overall_accuracy.keys())
plt.ylabel("Apmācības laiks (sekundes)")
plt.grid(axis='y')
plt.show()

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

rffpr = [0, 0, 0, 0.16666667 0.16666667 0.33333333, 0.33333333 0.5        0.83333333 0.83333333 1.        ]

plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# TODO: save ROC curve data
print(f"X fpr: {fpr}")
print(f"Y tpr: {tpr}")





# plot trainingduration
# plt.plot(rfData['trainingduration'], 'r', label='RandomForest')
# plt.plot(cnnData['trainingduration'], 'b', label='CNN')
# plt.plot(svmData['trainingduration'], 'g', label='SVM')
# plt.ylabel("Kopējā precizitāte")
# plt.xlabel("Klasifikatora numurs")
# plt.legend()
# plt.show()



# References:
# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
# https://www.statology.org/plot-roc-curve-python/

# https://www.youtube.com/watch?v=4jRBRDbJemM&ab_channel=StatQuestwithJoshStarmer








