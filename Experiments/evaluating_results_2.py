

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

# AVG overall accuracy
overallRFAccuracy = rfData['overallAccuracy']
overallCNNAccuracy = cnnData['overallAccuracy']
overallSVMAccuracy = svmData['overallAccuracy']

height = [mean(overallRFAccuracy), mean(overallCNNAccuracy), mean(overallSVMAccuracy)]
bars = ('RandomForest', 'CNN', 'SVM')
y_pos = np.arange(len(bars))
# Create bars
plt.bar(y_pos, height)
# Create names on the x-axis
plt.xticks(y_pos, bars)
# Show graphic
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








