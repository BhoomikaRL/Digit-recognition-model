import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

randBool, img = cap.read()
cap.release()
#%matplotlib auto
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

imgR = img[:,:,2]

plt.imshow(imgR,cmap='gray')
plt.show()
r,c,l = img.shape
imgOutput = np.ones((r,c)).astype('uint8')
imgOutput[:,0:100]=255
cv2.imshow('output',imgOutput)
cv2.waitKey(0)

while(True):
    ret,img = cap.read()
    
    cv2.imshow("output",img)
    if cv2.waitKey(1) == 27:
        break











import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

while(True):
    ret,img = cap.read()
    
    cv2.imshow('Frames',img)
    if cv2.waitKey(1) == 27:
        break










import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

ret,img = cap.read()
cap.release()

imgR = img[:,:,2]
imgG = img[:,:,1]
imgB = img[:,:,0]

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

rMin=104
rMax=176

gMin=24
gMax=77

bMin=22
bMax=131
r,c,l = img.shape

output = np.zeros((r,c))
for i in range(0,r):
    for j in range(0,c):
        if imgR[i,j] >= rMax and imgR[i,j] <= rMin and imgG[i,j] >= gMax and imgG[i,j] <= gMin and imgB[i,j] >= bMax and imgB[i,j] <= bMin:
            output[i,j] = 255
plt.subplot(1,2,1)
plt.imshow(output,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
