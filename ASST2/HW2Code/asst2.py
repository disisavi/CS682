import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from os import walk
import random


def get_histogram(image, channels):
    hist = cv2.calcHist([image], channels, None, [256], [0, 256])
    return hist


def get_histogramnp(image, bin):
    # print(max(image.ravel()))
    hist, bins = np.histogram(image, bin, [0, bin])
    return hist


def draw_histogram(hist, color):
    plt.plot(hist, color=color)
    plt.xlim([0, 256])


def show_color_histogram(image):
    for i, col in enumerate(['b', 'g', 'r']):
        hist = get_histogram(image, [i])
        draw_histogram(hist, color=col)
    plt.show()

user_x = user_y = 1
def show_image(image, location):
    def onmouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            redValue = image[y, x, 2]
            greenValue = image[y, x, 1]
            blueValue = image[y, x, 0]
            global  user_x,user_y
            user_x = x
            user_y = y
            intensityValue = (int(redValue) + int(greenValue) + int(blueValue)) / 3
            print("\n\n\nImage Stats\n[{X:", x, " Y:", y, "}, {R:", image[y, x, 2], " G:", image[y, x, 1], " B:",
                  image[y, x, 0],
                  "}, {I:", intensityValue, "]")
            window = cv2.getRectSubPix(image, (11, 11), (x, y))

            windowBlue = np.mean(window[:, :, 0])
            windowGreen = np.mean(window[:, :, 1])
            windowRed = np.mean(window[:, :, 2])
            windowMean = np.mean(window)

            windowSDBlue = np.std(window[:, :, 0])
            windowSDGreen = np.std(window[:, :, 1])
            windowSDRed = np.std(window[:, :, 2])
            windowSD = np.std(window)
            print("Window stats(11x11)\n[ {mean{R:", windowRed, ", G:", windowGreen, ", B:", windowBlue,
                  "}", windowMean, "}]")
            print("[ {SD{R:", windowSDRed, ", G:", windowSDGreen, ", B:", windowSDBlue,
                  "}", windowSD, "}]")

            i2 = cv2.imread(location)
            cv2.rectangle(i2, (user_x - 5, user_y - 6), (user_x + 6, user_y + 5), (255, 255, 255))    
            cv2.imshow('image', i2)        
            cv2.imshow('11*11', window)


    cv2.namedWindow('image')
    cv2.imshow('image', image)
    cv2.setMouseCallback('image', onmouse)
    cv2.waitKey()
    cv2.destroyAllWindows()

def problem1(location):
    todo = True
    while todo:
        images = []
        print("the available files to chose are ... ")
        for (dirpath, dirnames, filenames) in walk(location):
            images.extend(filenames)
            break
        print('\033[92m', filenames, '\033[0m')
        filename = input("The fileanme ? (please enter exit to exit problem 1) ")
        if filename.upper() == "exit".upper():
            break
        location = location + filename
        image = cv2.imread(location, cv2.IMREAD_COLOR)
        if not image is None:
            todo = False
            print("Orignal image dimension = ", image.shape)
            print("Image loaded successfuly...")
            print("Please check the histogram for selected image in the popped up window.")
            show_color_histogram(image)
            print("Please check the image and the 11x11 window in the popped up window.")
            print("image stats will be shown in the console below...")
            show_image(image, location)
        else:
            print(
                "No image found... Check the path in asst2.py before trying.\nBe sure the script is being executed in HW2code, as the images take their relative paths from there.")
            print("Please try again")


def getVideo(location):
    cap = cv2.VideoCapture(location)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Number of images ", length)
    return cap


def calcHistogram(video):
    colorHistogram = []
    if not video.isOpened():
        print("Error opening video stream or file")

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            frame = frame.astype(np.uint16)

            # All colors divided by 32 -- 2^5
            frame = np.right_shift(frame, 5)
            # Red is multiplied by 64
            frame[:, :, 2] = np.left_shift(frame[:, :, 2], 6)
            # green is multiplied by 8
            frame[:, :, 1] = np.left_shift(frame[:, :, 1], 3)

            frame1 = frame[:, :, 2] + frame[:, :, 1] + frame[:, :, 0]
            hist = get_histogramnp(frame1, 512)
            colorHistogram.append(hist)

        else:
            break

    # print("Number of images ", len(colorHistogram))
    return colorHistogram


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

    # var = []
    # for i in range(len(hist1)):
    #     if (hist1[i] + hist2[i]) >= 5:
    #         var.append((pow(hist1[i] - hist2[i], 2) / hist1[i] + hist2[i]))
    #
    # num = sum(var)
    # print(histNUM)
    # print(num)
    num = num / den
    # print(num)
    return num


def problem2(location):
    location = location + '/ST2MainHall4%03d.jpg'

    video = getVideo(location)
    print("Calculating the histogram for all the images....")
    colorHistogramRGB = calcHistogram(video)

    print("Please check the histogram for 5 random images from the video in the popped up window.")
    # Show 5 random histograms from the video
    for i in range(5):
        histRGB = colorHistogramRGB[random.randint(0, len(colorHistogramRGB) - 1)]
        plt.plot(histRGB, color="blue")
        plt.xlim([0, 512])
        plt.show()

    histogram_intersection_matrix = np.zeros((99, 99), dtype=np.int16)
    histogram_chisquare_matrix = np.zeros((99, 99), dtype=np.float64)
    for i in range(len(colorHistogramRGB)):
        for j in range(len(colorHistogramRGB)):
            histogram_intersection_matrix[i][j] = (histogram_intersection(colorHistogramRGB[i],
                                                                          colorHistogramRGB[j])) * 255
            histogram_chisquare_matrix[i][j] = histogram_chisquare(colorHistogramRGB[i], colorHistogramRGB[j])

    print("Please check the scaled images for HistIntersection and ChiSquare in the popped up window...")
    print("please note that In both plots, smaller value corresponds to high similarity")
    histogram_intersection_matrix = histogram_intersection_matrix - 255
    histogram_intersection_matrix = np.absolute(histogram_intersection_matrix)
    print("Intersection matrix after scaling \n", histogram_intersection_matrix)
    plt.imshow(histogram_intersection_matrix)
    plt.colorbar()
    plt.title("Intersectoin Matrix")
    plt.show()

    maxvalue = np.amax(histogram_chisquare_matrix)
    print(maxvalue)
    histogram_chisquare_matrix = np.multiply(histogram_chisquare_matrix, 255 / maxvalue, casting='unsafe')
    histogram_chisquare_matrix = histogram_chisquare_matrix.astype(np.uint8)
    print("Chisquare matrix after scaling\n", histogram_chisquare_matrix)

    plt.imshow(histogram_chisquare_matrix)
    plt.colorbar()
    plt.title("Chi square Matrix")
    plt.show()


def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    location = '../ImageSource/'
    print("Problem 1.1 Starts here")
    problem1(location)
    print("***********\nProblem 1.1 ends here")
    print("Problem 1.2 starts here")
    location = '../ImageSource/ST2MainHall4'
    problem2(location)


if __name__ == "__main__": main()
