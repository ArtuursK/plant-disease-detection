
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

        # add color masking to segment only leaves
        rgb_img = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2RGB)
        hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

        # Define the Color Range to be Detected
        #light_green = (40, 40, 40)
        #dark_green = (70, 255, 255)
        # use (40, 40,40) ~ (70, 255,255) in hsv to find the green
        light_green = (40, 40, 40)
        dark_green = (70, 255, 255)
        mask = cv2.inRange(hsv_img, light_green, dark_green)

        resultImg = cv2.bitwise_and(resizedImg, resizedImg, mask=mask)
        # Saving the image
        cv2.imwrite(outputPath + file, resultImg)


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
