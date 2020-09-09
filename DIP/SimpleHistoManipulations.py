#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic histogram values manipulations.

@author: ssklykov
"""
# %% Imports
import LoaderFile
import matplotlib.pyplot as plt
from skimage import io


# %% Testing of LoaderFile
img1 = LoaderFile.loadSampleImage("wrong identifier")
# plt.imshow(img1) # For testing
sample_img = LoaderFile.loadSampleImage()
plt.figure("Sample")
io.imshow(sample_img)  # Also works in collabaration with matplotlib library


# %% Testing the identifier sending
ident = "some wrong identifier"
matchStrs = lambda string1, string2: string1 == string2
attepmt = matchStrs("ident", "castle")
