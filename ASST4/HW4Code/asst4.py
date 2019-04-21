import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import copy

locations = ['../imageSource/e1.png','../imageSource/e2.jpg']
alphas = [0,180,360]

def edgeImage(image):
    blurImage = cv2.GaussianBlur(image,(5,5),0)
    sigma = 0.5
    v = np.median(image)
    lower = int(max(0, (1 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    detected_edge = cv2.Canny(blurImage, lower, upper)
    return detected_edge

def normalized(image):
    height,width = image.shape
    returnImage = np.zeros((height, width), dtype=np.uint8)
    maxvalue = np.amax(image)
    returnImage = np.multiply(image, 255 / maxvalue, casting='unsafe')
    print(np.amax(returnImage))
    return returnImage

def getanges(radian):
    radian = math.radians(alpha)
    cosine = math.cos(radian)
    sine = math.sin(radian)
    return sine, cosine

def radonTransform(image,alpha):
    height,width = image.shape
    returnImage = np.zeros((height, width), dtype=np.uint32)
    indices = np.nonzero(image)
    sine,cosine = getanges(alpha)
    returnImage[indices] = indices[0]*cosine + indices[1]*sine
    return normalized(returnImage)

def radonTransformpy(image,alpha):
    height,width = image.shape
    returnImage = np.zeros((height, width), dtype=np.uint32)
    sine,cosine = getanges(alpha)
    for (x, y), item in np.ndenumerate(image):
        if item > 0:
        #     print("{"+str(x)+","+str(y)+"} --> "+str(x*cosine))
            returnImage[x][y] = x*cosine + y*sine
    return normalized(returnImage)

for i,location in enumerate(locations):
    for alpha in alphas:
        print(i,alpha)
        image = cv2.imread(location)
        edge = edgeImage(image)
        transform = radonTransformpy(edge,alpha)
        plt.imshow(transform)
        plt.show()