
import cv2
import os

path = os.path.dirname(os.path.abspath(__file__)) + "/O_Healthy_IMG_2071.JPG"
img = cv2.imread(path)

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

# Define the Color Range to be Detected
light_green = (40, 40, 40)
dark_green = (70, 255, 255)
mask = cv2.inRange(hsv_img, light_green, dark_green)

result = cv2.bitwise_and(img, img, mask=mask)
cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_color_masked.jpg'), result)

