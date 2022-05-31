
import os
import cv2

mainPath = os.path.dirname(os.path.abspath(__file__))
inputPath = mainPath + "/UnhealthyLeaves/UnhealthyLeavesTrain/"
outputPath = mainPath + "/UnhealthyLeaves/UnhealthyLeavesTrainResized/"

for file in os.listdir(inputPath):
    fullImagePath = inputPath + "/" + file
    img = cv2.imread(fullImagePath, cv2.IMREAD_UNCHANGED)
    print('Original Image Dimensions : ', img.shape)
    scale_percent = 15 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    print('Resized Dimensions : ', resized.shape)
    # Saving the image
    cv2.imwrite(outputPath + file, resized)

