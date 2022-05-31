


import matplotlib as plt
import numpy as np
import cv2
import os, random
from os import listdir

#D:\RTU\Datorredze\test_implementation\HealthyLeaves\HealthyLeavesTest\O_Healthy_IMG_2071.JPG
path = os.path.dirname(os.path.abspath(__file__)) + "/O_Healthy_IMG_2071.JPG"
img = cv2.imread(path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
twoDimage = img.reshape((-1, 3))
twoDimage = np.float32(twoDimage)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
attempts = 10

ret,label,center=cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((img.shape))
cv2.imshow('sample.jpeg', result_image)
cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample.jpg'), result_image)
#cv2.waitKey(0)



# Reference
# https://machinelearningknowledge.ai/image-segmentation-in-python-opencv/

