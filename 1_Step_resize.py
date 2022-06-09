
import os
import cv2

def preprocess(inputPath, outputPath):
    print(f"Reading images from ${inputPath}")
    for file in os.listdir(inputPath):
        fullImagePath = inputPath + "/" + file
        img = cv2.imread(fullImagePath, cv2.IMREAD_UNCHANGED)
        print('Original Image Dimensions : ', img.shape)
        # resize image
        resizedImg = cv2.resize(img, (256, 256))
        # Saving the image
        cv2.imwrite(outputPath + file, resizedImg)


mainPath = os.path.dirname(os.path.abspath(__file__))

inputPath = mainPath + "/UnhealthyLeaves/UnhealthyLeavesTrain/"
outputPath = mainPath + "/UnhealthyLeaves/UnhealthyLeavesTrainPreprocessed/"
preprocess(inputPath, outputPath)

inputPath = mainPath + "/UnhealthyLeaves/UnhealthyLeavesTest/"
outputPath = mainPath + "/UnhealthyLeaves/UnhealthyLeavesTestPreprocessed/"
preprocess(inputPath, outputPath)

inputPath = mainPath + "/HealthyLeaves/HealthyLeavesTrain/"
outputPath = mainPath + "/HealthyLeaves/HealthyLeavesTrainPreprocessed/"
preprocess(inputPath, outputPath)

inputPath = mainPath + "/HealthyLeaves/HealthyLeavesTest/"
outputPath = mainPath + "/HealthyLeaves/HealthyLeavesTestPreprocessed/"
preprocess(inputPath, outputPath)




# References
# https://machinelearningknowledge.ai/image-segmentation-in-python-opencv/
