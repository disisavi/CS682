import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import os

location = '../ImageSource/ST2MainHall4/'
alphas = [45, 90, 0]


def edgeImage(image):
    blurImage = cv2.GaussianBlur(image, (5, 5), 0)
    sigma = 0.6
    v = np.median(image)
    lower = int(max(0, (1 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    detected_edge = cv2.Canny(blurImage, lower, upper)
    return detected_edge


def normalized(image):
    height, width = image.shape
    returnImage = np.zeros((height, width), dtype=np.uint8)
    maxvalue = np.amax(image)
    returnImage = np.multiply(image, 255 / maxvalue, casting='unsafe')
    #     print(np.amax(returnImage))
    return returnImage


def getanges(alpha):
    radian = math.radians(alpha)
    cosine = math.cos(radian)
    sine = math.sin(radian)
    return sine, cosine


def radonTransform(image, alpha):
    height, width = image.shape
    returnImage = np.zeros((height, width))
    indices = np.nonzero(image)
    sine, cosine = getanges(alpha)
    returnImage[indices] = indices[0] * cosine + indices[1] * sine
    
    return returnImage


def radonTransformpy(image, alpha):
#     print(image.shape)
    height, width = image.shape
    returnImage = np.zeros((height, width))
    sine, cosine = getanges(alpha)
    for (x, y), item in np.ndenumerate(image):
        if item > 0:
            #     print("{"+str(x)+","+str(y)+"} --> "+str(x*cosine))
            returnImage[x][y] = x * cosine + y * sine
#     print(np.amax(returnImage), np.amin(returnImage))
    
    frequency = {}
    for (x,y), value in np.ndenumerate(returnImage):
        if value != 0:
            frequency.setdefault(value,[]).append((x,y))

    frequencyDesc = {}
    for k in sorted(frequency, key=lambda k: len(frequency[k]), reverse=True):
            frequencyDesc.setdefault(k,len(frequency[k]))

    for i, (k,v) in enumerate(frequencyDesc.items()):
        # print("\t- > ",frequency[k])
        idx = np.array(frequency[k])
        image[idx[:,0],idx[:,1]] = 100
        if i > 2:
           break
    return (image)


def sobelList(edge):
    img_blur = cv2.blur(edge, (5, 5))
    tempList = [cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=5), cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=5)]
    return tempList

def imageGradientGray(images):
    returnList = []

    for i, edge in enumerate(images):
        # print(edge.shape)
        edge = cv2.blur(edge, (5, 5))
        a = sobelList(edge)
        angle = cv2.phase(a[0], a[1], angleInDegrees=True)
        m = np.sqrt(np.square(a[0]) + np.square(a[1]))

        max = np.amax(m)
        m = (m * 255) / max
        m = m.astype('uint8')

        sub_180 = angle > 180
        angle[sub_180] -= 180
        angle = np.rint(angle / 5)
        # print(np.amax(angle))

        hist = np.histogram(angle, 36, [1, 36])

        plt.plot(hist[0], color="blue")
        plt.show()
        returnList.append(hist[0])

    return returnList


def loadALlImages(location):
    files = [file for file in glob.glob(location + '*jpg')]
    files.sort()
    images = [cv2.imread(file) for file in files]
    images.append(cv2.imread('../ImageSource/e1.png'))
    images.append(cv2.imread('../ImageSource/e2.jpg'))
    return images


# Implementaion starts from here 
os.system('cls' if os.name == 'nt' else 'clear')
images = loadALlImages(location)
print("image, angle")
for i, image in enumerate(images):
    edge = edgeImage(image)
    plt.imshow(edge, cmap="gray")
    plt.show()
    for alpha in alphas:
        print(i, alpha)
        transform = radonTransformpy(edge, alpha)
        plt.imshow(transform)
        plt.colorbar()
        plt.show()
images = [cv2.cvtColor(image, cv2.cv2.COLOR_BGR2GRAY) for image in images]
imageGradientGray(images)

# TODO
#         1. Improve edge 
#         2. complete part 2 of the project 
#         3. have an understanding of part 1 of the project
