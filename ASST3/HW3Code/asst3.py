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
    returnList = [sobelList(edge) for edge in detected_edges]
    return returnList

def sobelList(edge):
    
    tempList = []
    tempList.append(cv2.Sobel(edge,cv2.CV_64F,1,0,ksize=5))
    tempList.append(cv2.Sobel(edge,cv2.CV_64F,0,1,ksize=5))

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
    blurImages = []
    for image in images:
        edge = np.copy(image)
        edge[:, :, 0] = cannyEdgeDetector(image[:, :, 0])
        sobelListBlue = sobelList(edge[:, :, 0])
        edge[:, :, 1] = cannyEdgeDetector(image[:, :, 1])
        sobelListGreen = sobelList(edge[:, :, 1])
        edge[:, :, 2] = cannyEdgeDetector(image[:, :, 2])
        sobelListRed = sobelList(edge[:, :, 2])
        blurImages.append([sobelListRed,sobelListGreen,sobelListBlue])

    return blurImages

def grayEdgeHistogram(grayEdges):
    angle = [cv2.phase(grayEdges[i][0],grayEdges[i][1],angleInDegrees=True) for i in range(len(grayEdges)) ]
    for i in range(len(angle)):
        print(np.amax(angle[i]))

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
    grayEdges = imageGradientGray(grayImages)
    print("gray image gradient calculation done")
    print("Taking image Gradient of Color images")
    colorEdges = cannyEdgesColor(images)
    print("Color image gradient calculation done")
    print("Gray edge histogram calculation begins")
    grayEdgeHistogram(grayEdges)



if __name__ == "__main__": main()
