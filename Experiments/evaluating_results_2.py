

from numpy import mean

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

randomForestResults = "backup/RF_results.csv"
cnnResults = "backup/CNN_results.csv"
svmResults = "backup/SVM_results.csv"

rfData = pd.read_csv(randomForestResults)
cnnData = pd.read_csv(cnnResults)
svmData = pd.read_csv(svmResults)

# AVG training duration
height = [mean(rfData['trainingduration']), mean(cnnData['trainingduration']), mean(svmData['trainingduration'])]
bars = ('RandomForest', 'CNN', 'SVM')
plt.barh(bars, height)
for index, value in enumerate(height):
    plt.text(value, index, str(round(value,2)))
plt.xlabel('Apmācības laiks s')
#plt.xlim([0, 100])
plt.grid(axis='x')
plt.show()


# AVG overall accuracy
height = [mean(rfData['overallAccuracy']), mean(cnnData['overallAccuracy']), mean(svmData['overallAccuracy'])]
bars = ('RandomForest', 'CNN', 'SVM')
plt.barh(bars, height)
for index, value in enumerate(height):
    plt.text(value, index, str(round(value,2)))
plt.xlabel('Kopējā precizitāte %')
plt.xlim([0, 100])
plt.grid(axis='x')
plt.show()





# ROC curve (receiver operating characteristic curve) is a graph
# showing the performance of a classification model at all classification thresholds.
# This curve plots two parameters: True Positive Rate. False Positive Rate.




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








