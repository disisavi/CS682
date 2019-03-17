import glob
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def loadALlImages(location):
    files = [file for file in glob.glob(location + '*jpg')]
    files.sort()
    images = [cv2.imread(file) for file in files]
    return images


def convertToGray(images):
    gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    return gray


def imageGradientGray(images):
    detected_edges = [cannyEdgeDetector(image) for image in images]
    returnList = []
    xlist = []
    ylist = []
    imageList = []
    for i, edge in enumerate(detected_edges):
        a = sobelList(edge)
        if i == 0 or i == 28 or i == 65 or i == 95:
            xlist.append(a[0])
            ylist.append(a[1])
            imageList.append(images[i])

        angle = cv2.phase(a[0], a[1], angleInDegrees=True) // 10
        hist = np.histogram(angle, 35, [0, 35])
        returnList.append(hist[0])

    for i, image in enumerate(imageList):
        showHistquiverandGradient(xlist[i], ylist[i], image)

    return returnList


def sobelList(edge):
    tempList = [cv2.Sobel(edge, cv2.CV_64F, 1, 0, ksize=5), cv2.Sobel(edge, cv2.CV_64F, 0, 1, ksize=5)]
    return tempList


def cannyEdgeDetector(image):
    sigma = 0.6
    img_blur = cv2.blur(image, (5, 5))
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    detected_edge = cv2.Canny(img_blur, lower, upper)
    return detected_edge


def imageGradientColor(images):
    hist = []
    xlist = []
    ylist = []
    imageList = []
    edgeList = []

    for i, image in enumerate(images):
        rEdge = cannyEdgeDetector(image[:, :, 2])
        gEdge = cannyEdgeDetector(image[:, :, 1])
        bEdge = cannyEdgeDetector(image[:, :, 0])
        r = sobelList(rEdge)
        b = sobelList(bEdge)
        g = sobelList(gEdge)

        x = b[0] + g[0] + r[0]
        y = b[1] + g[1] + r[1]
        if i == 0 or i == 28 or i == 65 or i == 95:
            xlist.append(x)
            ylist.append(y)
            imageList.append(image)
        m, angle = cv2.cartToPolar(x, y, angleInDegrees=True)
        angle = angle // 10
        histogram = np.histogram(angle, range(36), weights=m)

        hist.append(histogram[0])

    for i, image in enumerate(imageList):
        showHistquiverandGradient(xlist[i], ylist[i], image)
    return hist


def histogram_intersection(hist1, hist2):
    min = np.sum(np.minimum(hist1, hist2))
    max = np.sum(np.maximum(hist1, hist2))
    return min / max


def histogram_chisquare(hist1, hist2):
    hist1NP = hist1[hist1 + hist2 >= 5]
    hist2NP = hist2[hist1 + hist2 >= 5]

    histNUM = np.square(hist1NP - hist2NP)
    num = np.sum(histNUM, dtype='float64')
    den = np.sum(hist2NP + hist1NP)

    num = num / den

    return num


def showHistComparision(histogram, color):
    histogram_intersection_matrix = np.zeros((99, 99), dtype=np.int16)
    histogram_chisquare_matrix = np.zeros((99, 99), dtype=np.float64)
    for i in range(len(histogram)):
        for j in range(len(histogram)):
            histogram_intersection_matrix[i][j] = (histogram_intersection(histogram[i],
                                                                          histogram[j])) * 255
            histogram_chisquare_matrix[i][j] = histogram_chisquare(histogram[i], histogram[j])

    histogram_intersection_matrix = histogram_intersection_matrix - 255
    histogram_intersection_matrix = np.absolute(histogram_intersection_matrix)
    print("Intersection matrix after scaling \n", histogram_intersection_matrix)
    plt.imshow(histogram_intersection_matrix, cmap='autumn')
    plt.colorbar()
    plt.title("Intersectoin Matrix " + color)
    plt.show()

    maxvalue = np.amax(histogram_chisquare_matrix)
    print(maxvalue)
    histogram_chisquare_matrix = np.multiply(histogram_chisquare_matrix, 255 / maxvalue, casting='unsafe')
    histogram_chisquare_matrix = histogram_chisquare_matrix.astype(np.uint8)
    print("Chisquare matrix after scaling\n", histogram_chisquare_matrix)

    plt.imshow(histogram_chisquare_matrix, cmap='autumn')
    plt.colorbar()
    plt.title("Chi square Matrix " + color)
    plt.show()


def showHistquiverandGradient(u, v, image):
    w = image.shape
    x, y = np.mgrid[0:w[1]:500j, 0:w[0]:500j]
    skip = (slice(None, None, 10), slice(None, None, 10))
    x, y = x[skip], y[skip]
    u, v = u[skip].T, v[skip].T
    if len(w == 3 ):
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')
    plt.quiver(x, y, u, v, width=0.001, scale=0.02, scale_units="width")
    plt.show()


def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    location = '../ImageSource/ST2MainHall4/'
    print("Loading all the images from the source...")
    images = loadALlImages(location)
    print("Image loading complete... ")
    print("Converting to greyscale... ")
    grayImages = convertToGray(images)
    print("Gray image conversion done")
    print("Taking image Gradient of Gray images")
    grayHistogramList = imageGradientGray(grayImages)
    colorHistogramList = imageGradientColor(images)
    print("gray image gradient calculation done")
    showHistComparision(grayHistogramList, "Gray")
    showHistComparision(colorHistogramList, "Color")


if __name__ == "__main__": main()
