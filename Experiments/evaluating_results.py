
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import shapiro
from scipy.stats import ttest_ind
from numpy import std
from numpy import mean

randomForestResults = "RF_results.csv"
cnnResults = "CNN_results.csv"
svmResults = "SVM_results.csv"

rfData = pd.read_csv(randomForestResults)
cnnData = pd.read_csv(cnnResults)
svmData = pd.read_csv(svmResults)

# compare:
# CNN and RF
# CNN and SVM
# RF and SVM

# plot accuracy
plt.plot(rfData['overallAccuracy'], 'r', label='RandomForest')
plt.plot(cnnData['overallAccuracy'], 'b', label='CNN')
plt.plot(svmData['overallAccuracy'], 'g', label='SVM')
plt.ylabel("Kopējā precizitāte")
plt.xlabel("Klasifikatora numurs")
plt.legend()
plt.show()

def normalityTest(accuracyData):
    print(f"accuracy mean = {mean(accuracyData)}")
    print(f"accuracy std = {std(accuracyData)}")
    plt.hist(accuracyData, color = 'blue', edgecolor = 'black')
    plt.show()
    alpha = 0.05
    print("Shapiro-Wilk test for normality")
    stat, p = shapiro(accuracyData)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print('Sample looks Normaly distributed (Gaussian) (failed to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

print("CNN Normality test:")
normalityTest(cnnData['overallAccuracy'])
print("SVM Normality test:")
normalityTest(svmData['overallAccuracy'])
print("RandomForest Normality test:")
normalityTest(rfData['overallAccuracy'])





# T-Test compare
# CNN and SVM and RF
#H0: mean values are equal.
# If the Independent t-test results are significant (p-value very very small p<0,05)
# we can reject the null hypothesis in support of the alternative hypothesis (difference is statistically significant)

# CNN and RF
# accuracyTTestResult = ttest_ind(cnnData['overallAccuracy'], rfData['overallAccuracy'])
# print(accuracyTTestResult)
# if(accuracyTTestResult.pvalue < alpha):
#     print("CNN accuracy and RF accuracy difference is statistically significant")
# else:
#     print("CNN accuracy and RF accuracy difference is not statistically significant. Mean values are equal")
# print("accuracyTTestResult.pvalue: ", accuracyTTestResult.pvalue)





