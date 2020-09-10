#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load / handle sample images for DIP samples

@author: ssklykov
"""
# %% Imports
from skimage import io
from skimage.util import img_as_ubyte
import os
import numpy as np


# %% Get the absolute path to the default folder with sample images
def getResourceFolder(folder: str = "resources") -> str:
    """The aim of this function - to return the absolute path to resource folder with sample images.
    As default it returns the path to resources folder that should be in same directory as this helper script.
    Input:
        resource folder name: str
    Return:
        absolute path to that folder
    """
    absolute_path = ""
    if folder == "resources":
        absolute_path = os.path.join(os.getcwd(), folder)
    else:
        absolute_path = os.getcwd()  # This part will be implemented... At some point
        print("Path hasn't been constructed, the cwd returned")
    return absolute_path


# %% The key function for loading images
def loadSampleImage(identifier: str = "castle"):
    """This function helps to fast load images and return them as pixels arrays.
    Input:
        Available identifiers as strings: "castle", "castle_rgb".
    Return:
        scikit-image object.
    """
    # Some blank image for default return
    sample = np.zeros((10, 10), dtype='int')
    sample = img_as_ubyte(sample, force_copy=True)
    # Default image return
    if identifier == "castle":
        resources = getResourceFolder()
        imagePath = os.path.join(resources, "nesvizh_grey.jpg")
        sample_open = io.imread(imagePath, as_gray=True)
        sample = img_as_ubyte(sample_open)
    return sample
