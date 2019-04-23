import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import os
import copy

location = '../ImageSource/ST2MainHall4/'
alphas = [0, 90, 45]


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


def printlines(image, rho, theta, inversebit=False):
    radian = 0
    if inversebit:
        radian = math.radians(theta)
    else:
        radian = theta

    a = math.cos(radian)
    b = math.sin(radian)

    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    if inversebit:
        cv2.line(image, (y1, x1), (y2, x2), (0, 0, 255), 2)
    else:
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)


def getLinePoints(dvalueMatrix, image, count):
    image = image.astype(int)
    image[image == 254] = -1
    image[image > 0] = 75
    image[image == -1] = 254
    dValue = []
    frequency = {}
    for (x, y), value in np.ndenumerate(dvalueMatrix):
        if value != 0:
            frequency.setdefault(value, []).append((x, y))

    frequencyDesc = {}
    for k in sorted(frequency, key=lambda k: len(frequency[k]), reverse=True):
        frequencyDesc.setdefault(k, len(frequency[k]))

    for i, (k, v) in enumerate(frequencyDesc.items()):
        # print("\t- > ",frequency[k])
        if (frequencyDesc[k] > 200):
            # print(k, frequencyDesc[k])
            idx = np.array(frequency[k])
            image[idx[:, 0], idx[:, 1]] = 254
            dValue.append(k)
            if i > count:
                break
        else:
            break
    if not dValue:
        return image, None
    return image, dValue


def radonTransform(edgeImage, alpha, image):
    image = copy.copy(image)
    height, width = edgeImage.shape
    returnImage = np.zeros((height, width))
    indices = np.nonzero(edgeImage)
    sine, cosine = getanges(alpha)
    returnImage[indices] = indices[0] * cosine + indices[1] * sine
    hImage, dlist = getLinePoints(returnImage, edgeImage, 2)
    if dlist is not None:
        for d in dlist:
            printlines(image, d, alpha, True)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    return hImage


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
            e = radonTransform(e, angle,image)
        plt.plot(hist, color="black")
        plt.show()
        returnList.append(e)

    return returnList


def getHoughTransform(image):
    edge = edgeImage(image)
    height, width = edge.shape
    lines = cv2.HoughLines(edge, 1, np.pi / 180, (int)((height + width) / 9))
    if lines is not None:
        for line in lines:
            for rho, theta in line:
                printlines(image, rho, theta)

    cv2.imshow("i", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


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
        transform = radonTransform(edge, alpha, image)
        plt.imshow(transform, cmap='bone')
        plt.colorbar()
        plt.show()
grayimages = [cv2.cvtColor(image, cv2.cv2.COLOR_BGR2GRAY) for image in images]
lineImages = imageGradientGray(grayimages)
for image in lineImages:
    plt.imshow(image, cmap='bone')
    plt.colorbar()
    plt.show()

print("part 3")

for image in images:
    getHoughTransform(image)

# TODO later
#     1. make stuff faster, like list comprehension while sorting and stuff
#     2. Find ways to make get lines faster by using numpy
#     3. Make Part 2 better by having to see on all the lines in vicinity of an angle
