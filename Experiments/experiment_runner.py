
import os

numberOfExperiments = 30
# Random forest
# classifierLocation = "../RandomForest/rf_test_1.py"
# testResultFileLocation = "RF_results.csv"

# CNN
# classifierLocation = "../CNN/cnn_test_3.py"
# testResultFileLocation = "CNN_results.csv"

# SVM
classifierLocation = "../SVM/svm_test_1.py"
testResultFileLocation = "SVM_results.csv"


# clean previous data
# expFile = open(testResultFileLocation, "w")
# expFile.write("time,overallAccuracy,sensitivity,specificity,precision,npv,trainingduration\n")
# expFile.close()

# run experiments
for i in range(numberOfExperiments):
    print(f"Experiment {i} start")
    os.system("python " + classifierLocation)
    print(f"Experiment {i} end")


