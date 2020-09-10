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


# %% Testing of LoaderFile
img1 = LoaderFile.loadSampleImage("wrong identifier")
# plt.imshow(img1) # For testing
sample_img = LoaderFile.loadSampleImage()

# %% Displaying of an image using plt + io.imshow()


# %% Displaying of an image using skimage.viewer - CHECK CONSISTENCY IF IT"S NEEDED
# viewer.ImageViewer(sample_img).show()


# %% Negative function for U8 images
def negativeUbyteImage(img, test: bool = False):
    """
    Calculates negative of an input U8 image. "test" - boolean to display or not results of this operation.
    """
    from skimage.util import img_as_ubyte
    import matplotlib.pyplot as plt
    from skimage import io
    MAX_VAL = 255  # Default maximum value for U8 images
    (rows, cols) = img.shape
    if img.dtype != 'uint8':
        img = img_as_ubyte(img, force_copy=True)
    if test:
        print("The input image size is:", cols, "x", rows)
        print('The input image type:', img.dtype)
    negative_img = np.zeros((rows, cols), dtype='uint8')
    # Single core calculations
    for i in range(rows):
        for j in range(cols):
            negative_img[i, j] = MAX_VAL - img[i, j]
    negative_img = img_as_ubyte(negative_img)
    if test:
        plt.figure("Sample")
        io.imshow(sample_img)  # Also works in collabaration with matplotlib library
        plt.figure("Negative")
        io.imshow(negative_img)
    return negative_img


# %% Testing of written functions
negative = negativeUbyteImage(sample_img, test=True)


# %% Testing minor functionality (simple calculations)
ident = "some wrong identifier"
matchStrs = lambda string1, string2: string1 == string2
attepmt = matchStrs("ident", "castle")
a = np.array([[1, 3, 4], [2, 3, 4]])
(rows, cols) = a.shape
