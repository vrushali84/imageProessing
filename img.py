
# B. Please find the Python Problem Statement:


import numpy as np
import cv2
image = cv2.imread('i1.png')


## Read this image and find the most dominant chocolate colour

height, width, _ = np.shape(image)
avg_color_per_row = np.average(image, axis=0)
avg_colors = np.average(avg_color_per_row, axis=0)
print(f'avg_colors: {avg_colors}')
int_averages = np.array(avg_colors, dtype=np.uint8)
print(f'int_averages: {int_averages}')
average_image = np.zeros((height, width, 3), np.uint8)
average_image[:] = int_averages
cv2.imshow("Avg Color", np.hstack([image, average_image]))
cv2.waitKey(0)
## finding the most dominant color












# masking everything , except the dominant color 
original = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([62,71,113], dtype="uint8")
upper = np.array([63,72,116], dtype="uint8")
mask = cv2.inRange(image, lower, upper)

cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cv2.fillPoly(mask, cnts, (255,255,255))
result = cv2.bitwise_and(original,original,mask=mask)

cv2.imshow('mask', mask)
cv2.imshow('result', result)
cv2.waitKey()
