from typing import Dict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
import os
import copy
import random

enablePart1, enablePart2, enablePart3, enablePart4 = False,False, True, False
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
    x1 = int(x0 + 100000 * (-b))
    y1 = int(y0 + 100000 * (a))
    x2 = int(x0 - 100000 * (-b))
    y2 = int(y0 - 100000 * (a))
    if inversebit:
        cv2.line(image, (y1, x1), (y2, x2), (0, 0, 255), 1)
    else:
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)


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


def getlinesimage(image, cimage, iterator):
    image = copy.copy(image)
    image = cv2.blur(image, (5, 5))
    max = []

    a = sobelList(image)
    angle = cv2.phase(a[0], a[1], angleInDegrees=True)

    angle[angle > 179] -= 180

    angle = np.rint(angle / 5)
    hist, bins = np.histogram(angle, 36, [0, 36])

    hist, bins = hist.tolist(), bins.tolist()
    plt.plot(hist, color="black")
    plt.title("Part 2 --> Orientation Histogram for image " + str(iterator + 1))
    plt.savefig("../ImageSource/2H1_i" + str(iterator + 1) + ".png")
    plt.show()
    t = np.mean(hist)

    for h in hist:
        if h > t:
            max.append(h)
    edge = edgeImage(cimage)
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
    plt.title("Part 2 --> Lines detected for image " + str(iterator + 1))
    plt.savefig("../ImageSource/2LD1_i" + str(iterator) + ".png")
    return edge, rouList, alphaList, cimage


def getHoughTransform(image: np.ndarray, i: str) -> list:
    image = copy.copy(image)
    edge = edgeImage(image)
    height, width = edge.shape
    threshold = int((height + width) / 10)
    if threshold <= 300:
        threshold = 300
    lines = cv2.HoughLines(edge, 2, np.pi / 180, threshold)

    if lines is not None:
        for line in lines:
            for rho, theta in line:
                printlines(image, rho, theta)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Part 3 --> Hough Transform : Lines detected for image " + i)
    plt.savefig("../ImageSource/3LD1_i" + i + ".png")
    plt.show()
    return image, lines


def getHoughProbabilisticTransform(image: np.ndarray, i: str):
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
    plt.title("Part 3 --> Probabilistic Hough Transform : Lines detected for image " + i)
    plt.savefig("../ImageSource/3LD2_i" + i + ".png")
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
    rholist = []
    thetalist = []

    if lines is not None:
        for line in lines:
            for rho, theta in line:
                if theta in thetalist:
                    temprho = rholist[thetalist.index(theta)]
                    temprho.append(rho)
                else:
                    thetalist.append(theta)
                    rholist.append([rho])
    return rholist, thetalist


def findVanishingPoint(list_segmented_lines: list, image: np.ndarray, segmentedLines: bool, itemNumber: str) -> list:
    stringtoSave = "../ImageSource/4LD"
    title = "Part 4 --> Vanishing Points with"
    shape = (image.shape[1], image.shape[0])
    image = copy.copy(image)
    if segmentedLines:
        rho_list = list_segmented_lines[1]
        theta_list = list_segmented_lines[2]
        theta_list = np.radians(theta_list).tolist()
        stringtoSave += "1_i"
        title += " Radon Transform for image"

    else:
        rho_list, theta_list = segment_lines_by_angles(list_segmented_lines[1])
        stringtoSave += "2_i"
        title += " Hough Transform for image"

    stringtoSave += itemNumber + ".png"
    title += itemNumber
    intersectionPointsHistogram: Dict[tuple, int] = {}
    for i in range(len(theta_list)):
        for j in range(i + 1, len(theta_list)):
            if theta_list[i] == theta_list[j]:
                pass
            intersection_point(rho_list[i], theta_list[i], rho_list[j], theta_list[j], intersectionPointsHistogram,
                               shape)
    intersectionPointsHistogram = {k: intersectionPointsHistogram[k] for k in
                                   sorted(intersectionPointsHistogram,
                                          key=lambda k: intersectionPointsHistogram[k], reverse=True)[:3]}
    for point, freq in intersectionPointsHistogram.items():
        if freq > 2:
            # print("\t\t", point, freq)
            cv2.circle(image, point, 10, (255, 0, 0), thickness=2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.savefig(stringtoSave)
    plt.show()


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
if enablePart1:
    print("Part 1")
    print('alphas are (in Degrees)', alphas)
    print("\nimage, angle")
    for j, image in enumerate(images):
        edge = edgeImage(image)
        # plt.imshow(edge, cmap="gray")
        # plt.show()
        i = j + 1
        for alpha in alphas:
            print(i, '\t', alpha)
            highlightLine, drwawnLine, _, _ = radonTransform(edge, alpha, image)
            plt.imshow(cv2.cvtColor(drwawnLine, cv2.COLOR_BGR2RGB), cmap='bone')
            plt.title("Part 1 --> Line Detected for Alpha " + str(alpha))
            plt.savefig("../ImageSource/1LD1_" + str(i) + "_" + str(alpha) + ".png")
            plt.show()
            plt.imshow(highlightLine, cmap='bone')
            plt.title("Part 1 --> Points contributing Line for Alpha " + str(alpha))
            plt.savefig("../ImageSource/1PD1_" + str(i) + "_" + str(alpha) + ".png")
            plt.colorbar()
            plt.show()
    print("**********************\nFor Random Angles")
    print("image, angle")
    for j, image in enumerate(images):
        i = j + 1
        for k in range(2):
            alpha = random.randint(0, 360)
            print(i, '\t', alpha)
            highlightLine, drwawnLine, _, _ = radonTransform(edge, alpha, image)
            plt.imshow(cv2.cvtColor(drwawnLine, cv2.COLOR_BGR2RGB), cmap='bone')
            plt.title("Part 1 --> Line Detected for Random Alpha " + str(alpha))
            plt.savefig("../ImageSource/1LD2_" + str(k) + "_i" + str(i) + ".png")
            plt.colorbar()
            plt.show()
            plt.imshow(highlightLine, cmap='bone')
            plt.title("Part 1 --> Points contributing Line for Random Alpha " + str(alpha))
            plt.savefig("../ImageSource/1PD2_" + str(k) + "_i" + str(i) + ".png")
            plt.colorbar()
            plt.show()
if enablePart2:
    print("\n\n###############################")
    print("\nPart 2 ")
    grayimages = [cv2.cvtColor(image, cv2.cv2.COLOR_BGR2GRAY) for image in images]

    for i, image in enumerate(grayimages):
        print("For image ", i + 1)
        radon_transform_list.append(getlinesimage(image, images[i], i))
        plt.imshow(radon_transform_list[i][0], cmap='bone')
        plt.title("Part 2 --> Lines detected for image " + str(i + 1))
        plt.savefig("../ImageSource/2PD1_i" + str(i) + ".png")
        plt.colorbar()
        plt.show()

if enablePart3:
    print("\n\n###############################")
    print("\npart 3")
    print("Lines drawn from Hugh Transform")
    for i, image in enumerate(images):
        print("\timag   e", i)
        hough_transform_list.append(getHoughTransform(image, str(int(i + 1))))
    print("Lines drawn from Probabilistic Hough Transform", i)
    for i, image in enumerate(images):
        print("\timage", i)
        getHoughProbabilisticTransform(image, str(i + 1))

if enablePart4:
    print("\n\n###############################")
    print("part 4")
    print("1. Vanishing points Using Radon Transform")
    for j, wow in enumerate(radon_transform_list):
        i = j + 1
        print("\timage ", i)
        findVanishingPoint(wow, images[j], True, str(i))
    print("\n2. Vanishing points Using Hough Transform")
    for j, wow in enumerate(hough_transform_list):
        i = j + 1
        print("\timage ", i)
        y, x, _ = images[j].shape
        findVanishingPoint(wow, images[j], False, str(i))

# TODO
#     0. Find a way to find 3 maxima's instead of 2 --> Done
#     1. Check for a faster implementation of getlinepoints()
#     2. Please comment the method structures
#     3. Draw line from one end to another
#     4. Check theta_list value please
