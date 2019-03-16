import glob
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
    for edge in detected_edges:
        a = sobelList(edge)
        angle = cv2.phase(a[0], a[1], angleInDegrees=True)
        angle = angle // 10
        hist = np.histogram(angle, 35, [0, 35])
        returnList.append(hist[0])
    return returnList


def sobelList(edge):
    tempList = [cv2.Sobel(edge, cv2.CV_64F, 1, 0, ksize=5), cv2.Sobel(edge, cv2.CV_64F, 0, 1, ksize=5)]

    return tempList


def cannyEdgeDetector(image):
    sigma = 0.6
    img_blur = cv2.blur(image, (5, 5))
    v = np.median(image)
    lower = int(max(0, (1 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    detected_edge = cv2.Canny(img_blur, lower, upper)
    return detected_edge


def cannyEdgesColor(images):
    hist = []

    for image in images:
        rEdge = cannyEdgeDetector(image[:, :, 2])
        gEdge = cannyEdgeDetector(image[:, :, 1])
        bEdge = cannyEdgeDetector(image[:, :, 0])
        r = sobelList(rEdge)
        b = sobelList(bEdge)
        g = sobelList(gEdge)

        x = b[0] + g[0] + r[0]
        y = b[1] + g[1] + r[1]

        m, angle = cv2.cartToPolar(x, y, angleInDegrees=True)
        angle = angle // 10
        histogram = np.histogram(angle, range(36), weights=m)
        hist.append(histogram[0])
    return hist


def histogram_intersection(hist1, hist2):
    min = np.sum(np.minimum(hist1, hist2))
    max = np.sum(np.maximum(hist1, hist2))
    return min / max


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
    print("gray image gradient calculation done")
    histogram_intersection_matrix = np.zeros((99, 99), dtype=np.int16)
    for i in range(len(grayHistogramList)):
        for j in range(len(grayHistogramList)):
            histogram_intersection_matrix[i][j] = (histogram_intersection(grayHistogramList[i],
                                                                          grayHistogramList[j])) * 255

    histogram_intersection_matrix = histogram_intersection_matrix - 255
    histogram_intersection_matrix = np.absolute(histogram_intersection_matrix)
    print("Intersection matrix after scaling \n", histogram_intersection_matrix)
    plt.imshow(images[0])
    plt.colorbar()
    plt.title("Intersectoin Matrix")
    plt.show()

    colorHistogramList = cannyEdgesColor(images)
    histogram_intersection_matrix = np.zeros((99, 99), dtype=np.int16)
    for i in range(len(colorHistogramList)):
        for j in range(len(colorHistogramList)):
            histogram_intersection_matrix[i][j] = (histogram_intersection(colorHistogramList[i],
                                                                          colorHistogramList[j])) * 255

    histogram_intersection_matrix = histogram_intersection_matrix - 255
    histogram_intersection_matrix = np.absolute(histogram_intersection_matrix)
    print("Intersection matrix after scaling \n", histogram_intersection_matrix)
    plt.imshow(histogram_intersection_matrix)
    plt.colorbar()
    plt.title("Intersection Matrix")
    plt.show()


if __name__ == "__main__": main()
