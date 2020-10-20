#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing script of implemented classes / functions.

@author: ssklykov
"""
# %% Imports
import LoaderFile
import numpy as np
from wrapperClassUbyteImg import WrapUbyteImg as wrapImg

# %% Testing classes / functions

# %% Tests minorities features
sample_img = LoaderFile.loadSampleImage()
another_img = [[1, 2, 5], [2, 3, 6]]  # 2D list, still iterable
last_img = np.zeros((2, 2))  # Maybe, there are much more possible ways of image representation
transfer_img = np.array(another_img)
random_obj = 'just to test capabilities of shape feature'
print(str(type(sample_img)))
print(str(type(another_img)))
# print(dir(another_img))
shape = len(another_img)
print(str(type(last_img)))
print(str(type(transfer_img)))
if "ndarray" in str(type(last_img)):
    print("The type successfully checked")
try:
    print(random_obj.shape)
except AttributeError:
    print("This object doesn't have an attribute 'shape'")

img1 = wrapImg(sample_img)
img2 = wrapImg(another_img)
