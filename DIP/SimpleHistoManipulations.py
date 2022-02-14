#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic histogram values manipulations.

@author: ssklykov
"""
# %% Imports
import LoaderFile
# from skimage import viewer  # Not best solution, somehow causes freezing of a kernel in an infinite loop
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
# from skimage.util import img_as_uint  # It's smartly converting U8 to U16 by transfer 255 -> 65535 (max values)
from skimage import io
from numpy.random import randint

# %% Testing of LoaderFile
img1 = LoaderFile.loadSampleImage("wrong identifier")
# plt.imshow(img1) # For testing
sample_img = LoaderFile.loadSampleImage()

# Displaying of an image using plt + io.imshow() - in functions below


# %% Displaying of an image using skimage.viewer - CHECK CONSISTENCY IF IT"S NEEDED
# viewer.ImageViewer(sample_img).show()


# %% Calculation of negative for U8 images
def negativeUbyteImage(img, test: bool = False):
    """
    Calculate negative of an input U8 image usng slow pixelwise loops.

    Inputs:
        img - 2D image
        test - boolean flag to display or not results of this operation
    """
    MAX_VAL = 255  # Default maximum value for U8 images
    (rows, cols) = img.shape
    if img.dtype != 'uint8':
        img = img_as_ubyte(img, force_copy=True)
    if test:
        print("The input image size is:", cols, "x", rows)
        print('The input image type:', img.dtype)
    # Performance measure
    if test:
        # start_time = time.clock()  # Complains about deprecation
        start_time = time.perf_counter()
    else:
        start_time = 0
    negative_img = np.zeros((rows, cols), dtype='uint8')
    # Single core, pixelwise calculations
    for i in range(rows):
        for j in range(cols):
            negative_img[i, j] = MAX_VAL - img[i, j]
    negative_img = img_as_ubyte(negative_img)
    # Performance measure
    if test:
        finish_time = time.perf_counter()
        print("The algoritm takes to perform:", round(finish_time - start_time, 3), "s")
    else:
        finish_time = 0
    if test:
        plt.figure("Sample")
        io.imshow(img)  # Also works in collabaration with matplotlib library
        plt.figure("Negative")
        io.imshow(negative_img)
    return negative_img


def negativeVectorizedUbyteImg(img, test: bool = False):
    """Attempt to speed up the function above by using vectorization of substraction operation on an image.

    Inputs:
        img - 2D image
        test - boolean flag for displaying info about performed tests
    Returns:
        2D U8 negative to an input image
    """
    MAX_VAL = 255  # Default maximum value for U8 images
    (rows, cols) = img.shape
    # Performance measure
    if test:
        start_time = time.perf_counter()
    else:
        start_time = 0
    negative_img = np.zeros((rows, cols), dtype='uint8')
    # Vectorized operation
    negative_img = MAX_VAL + np.uint8(np.multiply(img, -1))  # really fast and simple operation over image
    negative_img = img_as_ubyte(negative_img)
    if test:
        finish_time = time.perf_counter()
        print("The algoritm takes to perform:", round(finish_time - start_time, 3), "s")
    else:
        finish_time = 0
    if test:
        # plt.figure("Sample")
        # io.imshow(img)  # Also works in collabaration with matplotlib library
        plt.figure("Negative")
        io.imshow(negative_img)
    return negative_img


# %% Log function calculation for U8 images
def logUbyteImage(img, test: bool = False, vectorized: bool = True, maxConstant: bool = True, c: float = 1.0):
    """Calculate results of linear logarithm application to image intensities.

    Input:
        img - image (preferablly U8)
        test - boolean for displaying test results or not
        c - constant of log conversion, will be checked for consistency (0 < c < max)
        vectorized - flag to use pixelwise or vectorized form of operation
    Output:
        U8 type image wit recalculated pixel values
    """
    start_time = time.perf_counter()
    (rows, cols) = img.shape
    if img.dtype != 'uint8':
        img = img_as_ubyte(img, force_copy=True)
    # if test:
    #     print("The input image size is:", cols, "x", rows)
    #     print('The input image type:', img.dtype)
    log_img = np.zeros((rows, cols), dtype='uint8')
    maxPixelInImg = np.max(img)
    cMax = np.log(maxPixelInImg + 1)
    cMax = 255/cMax  # Making maximum value of pixels equal to 255
    if c < 0 or c == 0.0:
        c = cMax
        print("Just warning: constant c - inconsistent, used one c =", c)
    elif c > cMax:
        c = cMax
        print("Just warning: constant c - inconsistent, used one c =", c)
    elif maxConstant:
        c = cMax
        print("Max constant used: ", round(c, 3))
    # Calculation of linear logarithm + conversion of pixel values
    if vectorized:
        shifted_img = np.uint16(img)  # Needed trick because 255 + 1 operation below automatically converts to 0 for U8 img
        shifted_img = np.add(shifted_img, 1)
        # print("Min pixel value in a shifted image:", np.min(shifted_img))
        log_img = np.uint8(np.multiply(c, np.log(shifted_img)))
    else:
        for i in range(rows):
            for j in range(cols):
                log_img[i, j] = np.uint8(c*np.log(1 + img[i, j]))
    log_img = img_as_ubyte(log_img)
    finish_time = time.perf_counter()
    if test:
        unique_id = randint(0, 101) + randint(0, 11)  # Dirty hack to avoid overwriting of results for multiple calls in test
        unique_id = str(unique_id)
        plt.figure("Sample" + unique_id)
        io.imshow(img)  # Also works in collabaration with matplotlib library
        plt.figure("c*ln(Sample)" + unique_id)
        io.imshow(log_img)
        print("The algoritm takes to perform:", round(finish_time - start_time, 3), "s")
    return log_img


# %% Gamma-correction of U8 images
def gammaUbyteImage(img, test: bool = False, a: float = 1.0, gamma: float = 1.0, vectorized: bool = True):
    """
    Apply gamma correction to pixel values in the form of new_pixel = a*((sample_pixel)**gamma).

    Input:
        a - coefficient (a > 0, will be assigned to 1 otherwise)
        gamma - power of transformation
    Returns:
        gamma-corrected U8 image
    """
    start_time = time.perf_counter()
    (rows, cols) = img.shape
    if img.dtype != 'uint8':
        img = img_as_ubyte(img, force_copy=True)
    gamma_img = np.zeros((rows, cols), dtype='uint8')
    maxPixelInImg = np.max(img)
    if abs(a) < 1e-6:  # approximation of float zero
        a = 1.0
        print("Warning: a can't be zero")
    cMax = a*np.power(maxPixelInImg, gamma)
    cMax = 255/cMax  # Making maximum value of pixels equal to 255
    if vectorized:
        coeff = np.multiply(cMax, a)
        # print("coeff:", coeff)
        interimFloatImg = np.float16(img)  # Again, avoid rounding in operation np.power(img, gamma)
        interimFloatImg = np.multiply(coeff, np.power(interimFloatImg, gamma))
        gamma_img = np.uint8(interimFloatImg)
    else:
        for i in range(rows):
            for j in range(cols):
                gamma_img[i, j] = np.uint8(cMax*a*np.power(img[i, j], gamma))
    # gamma_img = img_as_ubyte(gamma_img)
    finish_time = time.perf_counter()
    if test:
        print("The algoritm takes to perform:", round(finish_time - start_time, 3), "s")
        # unique_id = randint(0, 101) + randint(0, 11)  # Diry hack to avoid overwriting of results for multiple calls in test
        # unique_id = str(unique_id)
        unique_id = " a: " + str(a) + ", " + "gamma: " + str(gamma)
        # plt.figure("Sample" + unique_id)  # For comparison of parameters influence on the result
        # io.imshow(img)  # Also works in collabaration with matplotlib library
        plt.figure("a*(Sample**gamma) " + unique_id)
        io.imshow(gamma_img)
    return gamma_img


# %% Testing of written functions
# negative = negativeUbyteImage(sample_img, test=True)  # Testing of calculation of negative of an image
# negative2 = negativeVectorizedUbyteImg(sample_img, True)
logImage = logUbyteImage(sample_img, True)
# logImage2 = logUbyteImage(sample_img, True, False)
gammaImg = gammaUbyteImage(sample_img, True, 0.5, 2, True)
# gammaImg2 = gammaUbyteImage(sample_img, True, 0.5, 2, False)

# %% Testing minor functionality (simple calculations)
ident = "some wrong identifier"
matchStrs = lambda string1, string2: string1 == string2
attepmt = matchStrs("ident", "castle")
a = np.array([[1, 3, 4], [2, 3, 4]])
(rows, cols) = a.shape

# %% Close all redundant external displays after some waiting time (checking the results briefly)
# time.sleep(22)  # causes the errors during displaying of results
# input("Press enter to finish this script and close all displays ...")
# plt.close('all')
