import numpy as np
from sklearn.externals import joblib
import cv2
import matplotlib.pyplot as plt

classifier = joblib.load('digitsmodel')

image = cv2.imread('photo_1.jpg',0)

plt.imshow(image,cmap='gray')

blurred = cv2.GaussianBlur(image,(5,5),0)

plt.imshow(blurred,cmap="gray")

ret,thresholdImage = cv2.threshold(blurred,90,255,cv2.THRESHOLD_BINARY)

plt.imshow(thresholdImage,cmap="gray")

contours,_heir = cv2.findContours(thresholdImage,cv2.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(ctr) for ctr in ctrs]

rects = []
for ctr in contours:
    rects.append(cv2.boundingRect(ctr))

val1 = rects[2]
cv2.rectangle(image,(val1[0],val1[1]),(val1[0]+val1[2],(val1[1]+val1[3]),(0,255,0),2)

plt.imshow(image[val1[1]-10:val1[1]+val1[3]+10,val1[0]-10:val1[0]+10])

