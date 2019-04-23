import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import os

location = '../ImageSource/ST2MainHall4/'
alphas = [0, 45]


def edgeImage(image):
    blurImage = cv2.GaussianBlur(image, (5, 5), 0)
    sigma = 0.6
    v = np.median(image)
    lower = int(max(0, (1 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    detected_edge = cv2.Canny(blurImage, lower, upper)
    return detected_edge


def getanges(alpha):
    radian = math.radians(alpha)
    cosine = math.cos(radian)
    sine = math.sin(radian)
    return sine, cosine


def getLinePoints(dvalue, image, count):
    image = image.astype(int)
    image[image == 254] = -1
    image[image > 0] = 75
    image[image == -1] = 254
    frequency = {}
    for (x, y), value in np.ndenumerate(dvalue):
        if value != 0:
            frequency.setdefault(value, []).append((x, y))

    frequencyDesc = {}
    for k in sorted(frequency, key=lambda k: len(frequency[k]), reverse=True):
        frequencyDesc.setdefault(k, len(frequency[k]))

    for i, (k, v) in enumerate(frequencyDesc.items()):
        # print("\t- > ",frequency[k])
        idx = np.array(frequency[k])
        image[idx[:, 0], idx[:, 1]] = 254
        if i > count:
            break
    return image


def radonTransform(image, alpha):
    height, width = image.shape
    returnImage = np.zeros((height, width))
    indices = np.nonzero(image)
    sine, cosine = getanges(alpha)
    returnImage[indices] = indices[0] * cosine + indices[1] * sine

    return getLinePoints(returnImage, image, 10)


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

    return getLinePoints(returnImage, image, 10)


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

        sub_180 = angle > 180
        angle[sub_180] -= 180

        angle = np.rint(angle / 5)
        hist, bins = np.histogram(angle, 36, [0, 36])
        max = [-1, -1]
        hist, bins = hist.tolist(), bins.tolist()
        for h in hist:
            if h > max[0]:
                max[1] = max[0]
                max[0] = h
            elif h > max[1]:
                max[1] = h
        e = edgeImage(edge)
        for element in max:
            angle = bins[hist.index(element)] * 5
            print(angle)
            e = radonTransform(e, angle)
        plt.plot(hist, color="black")
        plt.show()
        returnList.append(e)

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
    # plt.imshow(edge, cmap="gray")
    # plt.show()
    for alpha in alphas:
        print(i, '\t', alpha)
        transform = radonTransformpy(edge, alpha)
        plt.imshow(transform, cmap='bone')
        plt.colorbar()
        plt.show()
images = [cv2.cvtColor(image, cv2.cv2.COLOR_BGR2GRAY) for image in images]
lineImages = imageGradientGray(images)
for image in lineImages:
    plt.imshow(image, cmap='bone')
    plt.colorbar()
    plt.show()
