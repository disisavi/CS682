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
    returnList = []
    xlist = []
    ylist = []
    imageList = []
    edgeList = []
    for i, edge in enumerate(images):
        a = sobelList(edge)
        angle = cv2.phase(a[0], a[1], angleInDegrees=True)
        m = np.sqrt(np.square(a[0]) + np.square(a[1]))
        angle, m = nonMaxSuppression(angle, m)
        hist = np.histogram(angle, 35, )
        if i == 0 or i == 28 or i == 65 or i == 95:
            xlist.append(a[0])
            ylist.append(a[1])
            imageList.append(images[i])
            edgeList.append(m)
        returnList.append(hist[0])

    for i, image in enumerate(imageList):
        showHistquiverandGradient(xlist[i], ylist[i], image, returnList[i], edgeList[i], "gray", i)

    return returnList


def imageGradientColor(images):
    hist = []
    xlist = []
    ylist = []
    imageList = []
    edgeList = []

    for i, image in enumerate(images):
        r = sobelList(image[:, :, 2])
        b = sobelList(image[:, :, 1])
        g = sobelList(image[:, :, 1])

        x = b[0] + g[0] + r[0]
        y = b[1] + g[1] + r[1]
        m, angle = cv2.cartToPolar(x, y, angleInDegrees=True)
        angle = angle // 10
        maxvalue = np.amax(m)
        m = np.multiply(m, 255) / maxvalue
        if i == 0 or i == 28 or i == 65 or i == 95:
            xlist.append(x / 3)
            ylist.append(y / 3)
            imageList.append(image)
            edgeList.append(m)
        histogram = np.histogram(angle, range(36))

        hist.append(histogram[0])

    for i, image in enumerate(imageList):
        showHistquiverandGradient(xlist[i], ylist[i], image, hist[i], edgeList[i], "Color", i)
    return hist


def nonMaxSuppression(angle, magnitude):

    return angle, magnitude


def sobelList(edge):
    img_blur = cv2.blur(edge, (5, 5))
    tempList = [cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=5), cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=5)]
    return tempList


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
    print("Intersection matrix for " + color + " matrix after scaling \n", histogram_intersection_matrix)
    plt.imshow(histogram_intersection_matrix)
    plt.colorbar()
    plt.title("Intersectoin Matrix " + color)
    plt.show()

    maxvalue = np.amax(histogram_chisquare_matrix)
    print(maxvalue)
    histogram_chisquare_matrix = np.multiply(histogram_chisquare_matrix, 255 / maxvalue, casting='unsafe')
    histogram_chisquare_matrix = histogram_chisquare_matrix.astype(np.uint8)
    print("Chisquare matrix for " + color + " after scaling\n", histogram_chisquare_matrix)

    plt.imshow(histogram_chisquare_matrix)
    plt.colorbar()
    plt.title("Chi square Matrix " + color)
    plt.show()


def showHistquiverandGradient(u, v, image, histogram, edge, code, index=0):
    location = '../ImageSource/'
    fig = plt.figure()
    fig.suptitle("Part 1 --- The  Sobel Gradient of the Image", fontsize=16)
    plt.subplot(131), plt.imshow(image), plt.title("Image")
    plt.subplot(132), plt.imshow(u, cmap="gray"), plt.title("X gradient")
    plt.subplot(133), plt.imshow(v, cmap="gray"), plt.title("Y gradient")
    # plt.savefig(location + code + " " + str(index) + " Sobel o/p.png")
    plt.show()

    fig = plt.figure()
    fig.suptitle("Edges ", fontsize=16)
    plt.subplot(121), plt.imshow(image), plt.title("Image")
    plt.subplot(122), plt.imshow(edge, cmap="gray"), plt.title(code + " Edges")
    # plt.savefig(location + code + " Edges.png")
    plt.show()

    plt.plot(histogram, color="blue")
    plt.title("Histogram for the " + code + " image")
    # plt.savefig(location + code + " Histogram.png")
    plt.show()

    w = image.shape
    x, y = np.mgrid[0:w[1]:500j, 0:w[0]:500j]
    skip = (slice(None, None, 10), slice(None, None, 10))
    x, y = x[skip], y[skip]
    u, v = u[skip].T, v[skip].T

    if len(w) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')

    plt.quiver(x, y, u, v, width=0.001, scale=0.0002, scale_units="width")
    # plt.savefig(location + code + " quiver.png")
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
    print("gray image gradient calculation done")
    print("Taking image Gradient of Color images")
    colorHistogramList = imageGradientColor(images)
    print("Color image gradient calculation done")
    showHistComparision(grayHistogramList, "Gray")
    showHistComparision(colorHistogramList, "Color")


if __name__ == "__main__": main()
