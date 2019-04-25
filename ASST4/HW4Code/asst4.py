import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import os
import copy

enablePart1, enablePart2, enablePart3, enablePart4 = False, True, True, True
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


def getLinePoints(dvalueMatrix, edgeImage, count):
    h, w = edgeImage.shape

    if int((h + w) / 20) > 100:
        threshold = int((h + w) / 20)
    else:
        threshold = 100
    edgeImage = edgeImage.astype(int)
    edgeImage[edgeImage == 254] = -1
    edgeImage[edgeImage > 0] = 75
    edgeImage[edgeImage == -1] = 254
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
        if frequencyDesc[k] > threshold:
            # print(k, frequencyDesc[k])
            idx = np.array(frequency[k])
            edgeImage[idx[:, 0], idx[:, 1]] = 254
            dValue.append(k)
        else:
            break

    if not dValue:
        edgeImage[0, 0] = 255
        return edgeImage, None

    return edgeImage, dValue


def radonTransform(edgeImage, alpha, image):
    image = copy.copy(image)
    height, width = edgeImage.shape
    returnImage = np.zeros((height, width))
    indices = np.nonzero(edgeImage)
    sine, cosine = getanges(alpha)
    returnImage[indices] = np.round((indices[0] * cosine + indices[1] * sine), 4)
    line_highlight, dlist = getLinePoints(returnImage, edgeImage, 2)
    if dlist is not None:
        for d in dlist:
            printlines(image, d, alpha, True)

    return line_highlight, image, dlist, alpha


def sobelList(edge):
    img_blur = cv2.blur(edge, (5, 5))
    tempList = [cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=1), cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=1)]
    return tempList


def imageGradientGray(image, cimage):
    image = copy.copy(image)
    image = cv2.blur(image, (5, 5))
    a = sobelList(image)
    angle = cv2.phase(a[0], a[1], angleInDegrees=True)

    angle[angle > 179] -= 180

    angle = np.rint(angle / 5)
    hist, bins = np.histogram(angle, 36, [0, 36])
    max = [-1, -1, -1]
    hist, bins = hist.tolist(), bins.tolist()

    plt.plot(hist, color="black")
    plt.show()

    max[0] = hist[0]
    for i in range(len(hist)):
        if i == 0:
            continue
        h = hist[i]
        if h > max[1]:
            max[2] = max[1]
            max[1] = h
        elif h > max[1]:
            max[2] = h

    edge = edgeImage(cimage)
    plt.imshow(edge)
    plt.show()
    rouList = []  # list of a list containing D for each alpha
    alphaList = []  # List of all the alphas for whcih i got a significant line
    for element in max:
        angle = bins[hist.index(element)] * 5
        print("\tangle selected is ", angle)
        for i in range(-3, 3):
            a = int(round(angle + i))
            print("\t\tChecking for ", int(angle), " + ", i)
            edge, cimage, rou, alpha = radonTransform(edge, a, cimage)
            if rou is not None:
                rouList.append(rou)
                alphaList.append(alpha)

    plt.imshow(cv2.cvtColor(cimage, cv2.COLOR_BGR2RGB))
    plt.show()
    return edge, rouList, alphaList


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
    return image, lines


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
    return image


def intersection_point(rho_list1: list, theta1: int or float, rho_list2: list, theta2: int or float,
                       intersectionPointsHistogram: dict, shape: tuple):
    angle_array = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    for rho1 in rho_list1:
        for rho2 in rho_list2:
            b = np.array([[rho1], [rho2]])
            x0, y0 = np.linalg.solve(angle_array, b)
            x0, y0 = int(np.round(x0)), int(np.round(y0))
            if x0 < shape[0] and y0 < shape[1]:
                f = intersectionPointsHistogram.setdefault((x0, y0), 1)
                intersectionPointsHistogram[(x0, y0)] = f + 1


def segment_lines_by_angles(lines):
    pass

def findVanishingPoint(list_segmented_lines: list, shape: tuple, segmentedLines: bool) -> list:
    if segmentedLines:
        printedLines = list_segmented_lines[0]
        rho_list = list_segmented_lines[1]
        theta_list = list_segmented_lines[2]
        intersectionPointsHistogram = {}
        for i in range(len(theta_list)):
            for j in range(i + 1, len(theta_list)):
                if theta_list[i] == theta_list[j]:
                    pass
                print(theta_list[i], theta_list[j])
                intersection_point(rho_list[i], theta_list[i], rho_list[j], theta_list[j], intersectionPointsHistogram,
                                   shape)
        for point, freq in intersectionPointsHistogram.items():
            if freq > 5:
                print("\t\t", point, freq)
    else:
        pass




def loadALlImages(location):
    files = [file for file in glob.glob(location + '*jpg')]
    files.sort()
    images = [cv2.imread(file) for file in files]

    return images


# Implementation starts from here
os.system('cls' if os.name == 'nt' else 'clear')
radon_transform_list = []  ## 0 --> Lines drawn 1. rou 2. Theta
hough_transform_list = []  ## 0 --> Lines drawn 2. unsegmented set of lines
images = loadALlImages(location)
if (enablePart1):
    print("Part 1")
    print("image, angle")
    for i, image in enumerate(images):
        edge = edgeImage(image)
        # plt.imshow(edge, cmap="gray")
        # plt.show()
        for alpha in alphas:
            print(i, '\t', alpha)
            highlightLine, drwawnLine, _, _ = radonTransform(edge, alpha, image)
            plt.imshow(cv2.cvtColor(drwawnLine, cv2.COLOR_BGR2RGB), cmap='bone')
            plt.colorbar()
            plt.show()
            plt.imshow(highlightLine, cmap='bone')
            plt.colorbar()
            plt.show()

if enablePart2:
    print("\n\nPart 2 ")
    grayimages = [cv2.cvtColor(image, cv2.cv2.COLOR_BGR2GRAY) for image in images]

    for i, image in enumerate(grayimages):
        print("For image ", i)
        radon_transform_list.append(imageGradientGray(image, images[i]))
        plt.imshow(radon_transform_list[i][0], cmap='bone')
        plt.colorbar()
        plt.show()

if enablePart3:
    print("part 3")
    print("Lines drawn from Hugh Transform")
    for i, image in enumerate(images):
        print("\timage", i)
        hough_transform_list.append(getHoughTransform(image))
    print("Lines drawn from Probabilistic Hough Transform", i)
    for i, image in enumerate(images):
        print("\timage", i)
        getHoughProbabilisticTransform(image)

if enablePart4:
    print("part 4")
    print("\t1. Vanishing points Using Radon Transform")
    for i, wow in enumerate(radon_transform_list):
        y, x, _ = images[i].shape

        findVanishingPoint(wow, (x, y), True)
    print("\t2. Vanishing points Using Hough Transform")
    for wow in hough_transform_list:
        y, x, _ = images[i].shape
        findVanishingPoint(wow, (x, y), False)

# TODO
#     0. Find a way to find 3 maxima's instead of2
#     1. Check for a faster implementation of getlinepoints()
#     2. Please comment the method structures
#     3. Draw line from one end to another
#     4. Check theta_list value please
