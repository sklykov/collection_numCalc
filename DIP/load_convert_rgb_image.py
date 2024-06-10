# -*- coding: utf-8 -*-
"""
Load stored RGB jpg image and convert it to the gray using various methods.

@author: sklykov
@license: The Unlicense

"""
# %% Imports
from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from pathlib import Path
import matplotlib.pyplot as plt
# import numpy as np
from skimage.exposure import histogram
from skimage.color import rgba2rgb

# %% Tests
# sample_img_path = Path(__file__).parent.joinpath("resources").joinpath("nesvizh.jpg")  # work as expected, no complains below
sample_img_path = Path(__file__).parent.joinpath("resources").joinpath("nesvizh.png")  # during the direct call, the 4 channel image opened
sample_img = None; sample_img_gray = None
tests = False

if sample_img_path.is_file():
    if tests:
        sample_img = io.imread(sample_img_path); img_shape = sample_img.shape
        if img_shape[2] == 3:  # for jpeg file
            sample_img_gray = rgb2gray(sample_img)  # RGB to float, 3 channels
            sample_img_gray = img_as_ubyte(sample_img_gray)
        elif img_shape[2] == 4:  # for png file
            sample_img_gray = io.imread(sample_img_path, as_gray=True)
            sample_img_gray = img_as_ubyte(sample_img_gray)
    else:
        # !!!: single line solution for conversion, working both for jpeg and png files
        sample_img_gray = img_as_ubyte(io.imread(sample_img_path, as_gray=True))

# Plotting the result
if sample_img_gray is not None:
    plt.close('all'); plt.figure(); plt.imshow(sample_img_gray, cmap=plt.cm.gray); plt.axis('off')
    plt.colorbar(); plt.tight_layout()

# Histogram for color image
if sample_img_path.is_file():
    sample_img = io.imread(sample_img_path); img_shape = sample_img.shape
    if img_shape[2] == 4:  # for png file
        sample_img = rgba2rgb(sample_img)
    plt.figure(); plt.imshow(sample_img); plt.axis('off'); plt.tight_layout()
    # Histogram - ...
