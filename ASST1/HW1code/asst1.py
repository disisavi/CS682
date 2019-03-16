import cv2
import numpy as np
import matplotlib.pyplot as plt

COLOR = "color"
GRAY = "grayscale"
Gnumber = 8


def show_image_plot(or_image, mod_image, code):
    or_image = cv2.cvtColor(or_image, cv2.COLOR_BGR2RGB)
    mod_image = cv2.cvtColor(mod_image, cv2.COLOR_BGR2RGB)
    plt.subplot(121), plt.imshow(or_image), plt.title('Input ' + code)
    plt.subplot(122), plt.imshow(mod_image), plt.title('Output')
    plt.show()


def img_perspective(image, code):
    pt = [[430, 150]]
    pt.append([900, 150])
    pt.append([430, 600])
    pt.append([900, 600])
    pts1 = np.float32([pt[0], pt[1], pt[2], pt[3]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    returnImage = cv2.warpPerspective(image, M, (300, 300))
    show_image_plot(image, returnImage, code + " perspective")

    return returnImage


def img_affine(image, code):
    val = image.shape
    pt = [[430, 150]]
    pt.append([900, 150])
    pt.append([430, 600])
    pts1 = np.float32([pt[0], pt[1], pt[2]])
    pts2 = np.float32([[420, 150], [930, 150], [500, 600]])

    M = cv2.getAffineTransform(pts1, pts2)
    returnimage = cv2.warpAffine(image, M, (val[0], val[1]))

    show_image_plot(image, returnimage, code + " affine")
    return returnimage


def img_erosion(image, code):
    kernel = np.ones((4, 4), np.uint8)
    returnimage = cv2.erode(image, kernel, iterations=1)
    show_image_plot(image, returnimage, code + " erosion")
    return returnimage


def img_convolution(image, code):
    kernel = np.ones((10, 10), np.float32) / 25
    returnimage = cv2.filter2D(image, -1, kernel)
    show_image_plot(image, returnimage, code + " 2Dconvolution")
    return returnimage


def img_bilfiltering(image, code):
    returnimage = cv2.bilateralFilter(image, 10, 100, 100)
    show_image_plot(image, returnimage, code + " Bil Filtering")
    return returnimage


def convert_greyscale(image):
    greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('../image/greyscale_convert.jpg', greyscale)
    print("A greyscale image has been saved in the images folder.. ")
    cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
    cv2.imshow('img1', greyscale)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return greyscale


def image_transformation(image, code):
    print("Image transformations started for the " + code + " photo")
    perspective = img_perspective(image, code)
    cv2.imwrite('../image/' + code + '_perspective.jpg', perspective)
    affine = img_affine(image, code)
    cv2.imwrite('../image/' + code + '_affine.jpg', affine)
    erosion = img_erosion(image, code)
    cv2.imwrite('../image/' + code + '_erosion.jpg', erosion)
    convolution = img_convolution(image, code)
    cv2.imwrite('../image/' + code + '_2dconvolution.jpg', convolution)
    bilfilter = img_bilfiltering(image, code)
    cv2.imwrite('../image/' + code + '_bil_filter.jpg', bilfilter)


def pyramid_gaussian(image):
    gaussian_list = []
    gaussian_list.append(image)
    maxy, maxx, temp = image.shape
    print("THe image dimension of each level of pyramid are as follows.")
    for n in range(Gnumber + 1):
        tempimage = cv2.pyrDown(gaussian_list[n])
        gaussian_list.append(tempimage)
        print(tempimage.shape)
    for n in range(Gnumber + 1):
        if Gnumber < 5:
            number = 100 + (Gnumber + 1) * 10 + n + 1
        else:
            if Gnumber % 2 == 0:
                number = 200 + (Gnumber + 2) * 5 + n + 1
            else:
                number = 200 + (Gnumber + 3) * 5 + n + 1
        plt.subplot(number), plt.imshow(cv2.cvtColor(gaussian_list[n], cv2.COLOR_BGR2RGB)), plt.title("Level " + str(n))
    plt.show()
    maxx += gaussian_list[1].shape[1]
    # Packing of image start here 
    packed_image = np.zeros((maxy, maxx, 3), dtype=np.uint8)
    packed_image.fill(255)
    n = x = y = 0
    for gimage in gaussian_list:
        y1, x1, temp = gimage.shape
        n += 1
        packed_image[y:y1 + y, x:x1 + x, ] = gimage
        if n == 2:
            n = 0
            y = y1 + y
        else:
            x = x1 + x
    print("Pyramid saved as StackedPyramid")
    cv2.imwrite('../image/StackedPyramid.jpg', packed_image)
    cv2.namedWindow('stack', cv2.WINDOW_NORMAL)
    cv2.imshow('stack', packed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    location = '../image/selfimage.jpg'
    image = cv2.imread(location, cv2.IMREAD_COLOR)
    if not image is None:
        print("Orignal image dimension = ", image.shape)
        print("Image loaded successfuly... Proceding to convet the image into greyscale")
        greyscale = convert_greyscale(image)
        print("The following conversion will be done on both image and greyscale... it will be saved in image directory\n1.Perspective Transformation\n2.Affine Transformation\n3.Erosion\n4.2D Convolution\n5.Bilateral Filtering")
        image_transformation(image, COLOR)
        print("Color Image transformation over.... Geryscale image transformation begins...")
        image_transformation(greyscale, GRAY)
        print("greyscale image transformation over")
        print("****************\nGausain pyramid\nWe will first go " + str(
            Gnumber) + " levels down and then Pack them together in a seperate step. \nPacked image will be saved in image folder ")
        pyramid_gaussian(image)
    else:
        print(
            "No image found... Check the path in asst1.py before trying.\nBe sure the script is being executed in HW1code, as the images take their relative paths from there.")


if __name__ == "__main__": main()
