# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 2019
'Getting started' with scikit-image library
scikit-image demo: open RGB image / translate it to uint8 gray image / get its normalized histogram
@author: ssklykov
"""
# %% Maybe for future (TODO): Specifying dependecies outside the file
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage import io
import skimage.exposure as exs
import os
import matplotlib.pyplot as plt
import numpy as np

# %% Path specifying to the sample image
currentPath = os.getcwd()  # returned current path
osInfo = os.uname()  # Infor in the format of a list with strings
osName = osInfo[0]
if "Win" in osName:
    pathToSample = "\\resources\\nesvizh.jpg"  # path to the sample picture. Windows-type path!
elif "nux" in osName:
    pathToSample = "/resources/nesvizh.jpg"
overallPath = currentPath + pathToSample
pathToResources = os.path.join(os.getcwd(), "resources")
# print(overallPath)  # Debugging

# %% Open and display the images
sample = io.imread(overallPath)
# plt.figure(1)
# plt.imshow(sample)  # Show the original image (suppresed - too many images)
graySample = img_as_ubyte(rgb2gray(sample))  # Two step conversion: RGB -> float_img -> uint8 image
# io.imsave(os.path.join(pathToResources, "nesvizh_grey.jpg"), graySample)
plt.figure(2)
plt.imshow(graySample, cmap=plt.cm.gray)  # The tip from the scikit-image webpage

# %% Calculation of histogram
hist = exs.histogram(graySample)  # get image histogram as 2 tuples

# %% Histogram normalization
maxCount = max(hist[0])  # Max count
countsNorm = np.asarray(hist[0], 'float')  # Conversion to np.array
for i in range(len(countsNorm)):
    countsNorm[i] /= maxCount

# %% Plot histogram
plt.figure(3)
axes = plt.axes()  # get the instance of axes for making minor plots and suppress the poped warning
plt.rc('font', family='serif')  # setting font type
plt.plot(hist[1], countsNorm, 'bo-', markersize=6, linewidth=2)
# %% Axes settings
plt.axis([min(hist[1]), max(hist[1]), 0, 1+0.05])  # ymax and ymin - histogram normalized
xTicks = np.arange(0, 257, 16, 'uint16')  # xMinorTicks = np.arange(0,257-16,16,'uint8')
plt.xticks(xTicks, fontsize=12, fontfamily='Liberation Serif')
yTicks = np.arange(0, 1.05, 0.1, 'float')
plt.yticks(yTicks, fontsize=12, fontfamily='Liberation Serif')
plt.xlabel('intensity values U8 gray image', fontsize=14, fontfamily='Liberation Serif')
plt.ylabel('normalized counts', fontsize=14, fontfamily='Liberation Serif')
# axes.set_xticks(xMinorTicks,minor=True)
plt.grid(which='both')
# %% Picture preparition and saving
plt.tight_layout()  # fill the picture better
