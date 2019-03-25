import asst3 as ast
import numpy as np
import cv2
import os

# check scaling for Vector 
# imagesc
# thresholding... 
def imageGradientColor(images):
    v = []
    xlist = []
    ylist = []
    imageList = []
    edgeList = []
    for i, image in enumerate(images):
        r = ast.sobelList(image[:, :, 2])
        b = ast.sobelList(image[:, :, 0])
        g = ast.sobelList(image[:, :, 1])
        x2 = b[0] + g[0] + r[0]
        y = b[1] + g[1] + r[1]

        a = np.square(r[0]) + np.square(g[0]) + np.square(b[0])
        c = np.square(r[1]) + np.square(g[1]) + np.square(b[1])
        b = np.multiply(r[0], r[1]) + np.multiply(g[0], g[1]) + np.multiply(b[0], b[1])
        ct2 = np.around(np.sqrt(np.square(a + c) - 4 * (np.square(b) - np.multiply(a, c))), 5)
        ct1 = np.around(a + c, 5)

        L1 = (ct1 + ct2) / 2
        L2 = (ct1 - ct2) / 2
        if np.count_nonzero(L1) == 0:
            lam = L1
        else:
            lam = L2

        lam = np.absolute(lam)
        m = np.sqrt(lam)
        # m = lam
        x = np.divide(b, a - lam, out=b, where=a - lam != 0)
        den = np.sqrt(np.square(x) + 1)
        e = [np.divide(x, den),
             1 / den]
        angle = cv2.phase(e[0], e[1], angleInDegrees=True)//5
        maxvalue = np.amax(m)
        m = np.multiply(m, 255) / maxvalue
        histogram = np.histogram(angle, range(36), weights=m)
        if i == 0 or i == 28 or i == 65 or i == 95:
            xlist.append(x2)
            ylist.append(y)
            imageList.append(images[i])
            edgeList.append(m)
        v.append(histogram[0])

    for i, image in enumerate(imageList):
        ast.showHistquiverandGradient(xlist[i], ylist[i], image, v[i], edgeList[i], "Eigenvectors Color")
    return v


def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    location = '../ImageSource/ST2MainHall4/'
    print("Loading all the images from the source...")
    print("Only Color Image will be processed for Euginvector")
    images = ast.loadALlImages(location)
    print("Image loading complete... ")

    print("Taking image Gradient of Color images")
    colorHistogramList = imageGradientColor(images)
    print("Color image gradient calculation done")

    ast.showHistComparision(colorHistogramList, "EV Color")


if __name__ == "__main__": main()
