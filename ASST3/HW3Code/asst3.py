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
    angleList = []
    for i, edge in enumerate(images):
        edge = cv2.blur(edge, (5, 5))
        a = sobelList(edge)
        angle = cv2.phase(a[0], a[1], angleInDegrees=True)
        m = np.sqrt(np.square(a[0]) + np.square(a[1]))

        max = np.amax(m)
        m = (m * 255) / max
        m = m.astype('uint8')
        if i == 0 or i == 28 or i == 65 or i == 95:
            xlist.append(a[0])
            ylist.append(a[1])
            imageList.append(images[i])
            edgeList.append(m)
            angleList.append(angle)
        
        angle = np.rint(angle/10)
        hist = np.histogram(angle,36,[1,36])
        returnList.append(hist[0])

    for i, image in enumerate(imageList):
        showHistquiverandGradient(xlist[i], ylist[i], image, returnList[i], edgeList[i], "gray", i,angleList[i])

    return returnList


def imageGradientColor(images):
    hist = []
    xlist = []
    ylist = []
    imageList = []
    edgeList = []
    angleList = []

    for i, image in enumerate(images):
        image = cv2.blur(image, (5, 5))
        r = sobelList(image[:, :, 2])
        b = sobelList(image[:, :, 1])
        g = sobelList(image[:, :, 1])

        x = b[0] + g[0] + r[0]
        y = b[1] + g[1] + r[1]
        m, angle = cv2.cartToPolar(x, y, angleInDegrees=True)
        # m = thresholding(angle, m)
        max = np.amax(m)
        m = (m*255)/max
        if i == 0 or i == 28 or i == 65 or i == 95:
            xlist.append(x / 3)
            ylist.append(y / 3)
            imageList.append(image)
            edgeList.append(m)
            angleList.append(angle)
        
        angle = np.rint(angle / 10)
        histogram = np.histogram(angle,36,[1,36])

        hist.append(histogram[0])

    for i, image in enumerate(imageList):
        showHistquiverandGradient(xlist[i], ylist[i], image, hist[i], edgeList[i], "Color", i,angleList[i])
    return hist


i = 0


def nonMaxSuppression(angle, magnitude):
    global i
    i += 1
    print("Non maxima suppression starts ", str(i))
    ymax, xmax = angle.shape
    m = np.zeros(angle.shape)
    for (x, y), item in np.ndenumerate(angle):
        if x > 0 and y > 0 and y < ymax and x < xmax:
            if (angle[x][y] >= 337.5 or angle[x][y] < 22.5) or (angle[x][y] >= 157.5 and angle[x][y] < 202.5):
                if magnitude[x][y] >= magnitude[x][y + 1] and magnitude[x][y] >= magnitude[x][y - 1]:
                    m[x][y] = magnitude[x][y]
            # 45 degrees
            if (angle[x][y] >= 22.5 and angle[x][y] < 67.5) or (angle[x][y] >= 202.5 and angle[x][y] < 247.5):
                if magnitude[x][y] >= magnitude[x - 1][y + 1] and magnitude[x][y] >= magnitude[x + 1][x - 1]:
                    m[x][y] = magnitude[x][y]
            # 90 degrees
            if (angle[x][y] >= 67.5 and angle[x][y] < 112.5) or (angle[x][y] >= 247.5 and angle[x][y] < 292.5):
                if magnitude[x][y] >= magnitude[x - 1][y] and magnitude[x][y] >= magnitude[x + 1][y]:
                    m[x][y] = magnitude[x][y]
            # 135 degrees
            if (angle[x][y] >= 112.5 and angle[x][y] < 157.5) or (angle[x][y] >= 292.5 and angle[x][y] < 337.5):
                if magnitude[x][y] >= magnitude[x - 1][y - 1] and magnitude[x][y] >= magnitude[x + 1][y + 1]:
                    m[x][y] = magnitude[x][y]
    print("Non maxima suppression Ends")
    return  m


def thresholding(angle, img):
    highThreshold = np.amax(img) * 0.8
    lowThreshold = highThreshold * 0.05
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(100)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img < highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res


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


def showHistquiverandGradient(u, v, image, histogram, edge, code, index,angle):
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
    x, y = x[skip].T, y[skip].T
    u, v = u[skip].T, v[skip].T
    u = u/10
    v = v/10
    if len(w) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image, cmap='gray')

    plt.quiver(x,y, u,v,width=0.01)
    
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
