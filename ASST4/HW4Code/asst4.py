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
    radian = -1
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
    h, w = image.shape
    threshold = -1
    if int((h + w) / 20) > 100:
        threshold = int((h + w) / 20)
    else : threshold = 100
    int((h + w) / 20)
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
        if (frequencyDesc[k] > threshold):
            # print(k, frequencyDesc[k])
            idx = np.array(frequency[k])
            image[idx[:, 0], idx[:, 1]] = 254
            dValue.append(k)
        else:
            break

    if not dValue:
        image[0, 0] = 255
        return image, None
    return image, dValue


def radonTransform(edgeImage, alpha, image):
    image = copy.copy(image)
    height, width = edgeImage.shape
    returnImage = np.zeros((height, width))
    indices = np.nonzero(edgeImage)
    sine, cosine = getanges(alpha)
    returnImage[indices] = np.round((indices[0] * cosine + indices[1] * sine),4)
    hImage, dlist = getLinePoints(returnImage, edgeImage, 2)
    if dlist is not None:
        for d in dlist:
            printlines(image, d, alpha, True)

    return hImage, image


def sobelList(edge):
    img_blur = cv2.blur(edge, (5, 5))
    tempList = [cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=5), cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=5)]
    return tempList


def imageGradientGray(image, cimage):
    image = copy.copy(image)
    image = cv2.blur(image, (5, 5))
    a = sobelList(image)
    angle = cv2.phase(a[0], a[1], angleInDegrees=True)

    angle[angle > 179] -= 180

    angle = np.rint(angle / 5)
    hist, bins = np.histogram(angle, 36, [0, 36])
    max = [-1, -1]
    hist, bins = hist.tolist(), bins.tolist()
    plt.plot(hist, color="black")
    plt.show()
    for h in hist:
        if h > max[0]:
            max[1] = max[0]
            max[0] = h
        elif h > max[1]:
            max[1] = h
    edge = edgeImage(image)
    cimagecopy = copy.copy(cimage)
    for element in max:
        angle = bins[hist.index(element)] * 5
        print("\tangle selected is ", angle)
        cimage = copy.copy(cimagecopy)
        for i in range(-3, 6):
            a = round(angle + i)
            print("\t\tChecking for ", angle, " + ", i)
            edge, cimage = radonTransform(edge, a, cimage)
        plt.imshow(cv2.cvtColor(cimage, cv2.COLOR_BGR2RGB))
        plt.show()
    return edge


def getHoughTransform(image):
    image = copy.copy(image)
    edge = edgeImage(image)
    height, width = edge.shape
    threshold = int((height + width) / 10)
    # print(threshold)
    lines = cv2.HoughLines(edge, 1, np.pi / 180, threshold)
    if lines is not None:
        for line in lines:
            for rho, theta in line:
                printlines(image, rho, theta)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def getHoughProbabilisticTransform(image):
    image = copy.copy(image)
    edge = edgeImage(image)
    minLineLength = 100
    maxLineGap = 50
    lines = cv2.HoughLinesP(edge, 1, np.pi / 90, 100, minLineLength, maxLineGap)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


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
print("Part 1")
print("image, angle")
for i, image in enumerate(images):
    edge = edgeImage(image)
    # plt.imshow(edge, cmap="gray")
    # plt.show()
    for alpha in alphas:
        print(i, '\t', alpha)
        highlightLine, drwawnLine = radonTransform(edge, alpha, image)
        plt.imshow(cv2.cvtColor(drwawnLine, cv2.COLOR_BGR2RGB), cmap='bone')
        plt.colorbar()
        plt.show()
        plt.imshow(highlightLine, cmap='bone')
        plt.colorbar()
        plt.show()

print("\n\nPart 2 ")
grayimages = [cv2.cvtColor(image, cv2.cv2.COLOR_BGR2GRAY) for image in images]

for i, image in enumerate(grayimages):
    print("For image ", i)
    plt.imshow(imageGradientGray(image, images[i]), cmap='bone')
    plt.colorbar()
    plt.show()

print("part 3")
print("Lines drawn from Hugh Transform")
for i, image in enumerate(images):
    print("\timage", i)
    getHoughTransform(image)
print("Lines drawn from Probabilistic Hough Transform", i)
for i, image in enumerate(images):
    print("\timage", i)
    getHoughProbabilisticTransform(image)

# TODO
#     1. For part 2, we need to print all the lines together. It is needed for vanishing points.
#     2. For part 2, there are a few quirks with getLinePoints(). Check for that and debug that please.
#     3. Check for a faster implementation of getlinepoints()
#     4. Check for better histogram Calculations for imageGradientGray()