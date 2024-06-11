# -*- coding: utf-8 -*-
"""
Load stored RGB jpg image and convert it to the gray using various methods.

Also, get the histogram for channels and transform to the various color schemes.

@author: sklykov
@license: The Unlicense

"""
# %% Imports
from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
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

# Plotting the result of RGB to gray-scaled image conversion
if sample_img_gray is not None:
    plt.close('all'); plt.figure(); plt.imshow(sample_img_gray, cmap=plt.cm.gray); plt.axis('off')
    plt.colorbar(); plt.tight_layout()

# Histogram for color image
if sample_img_path.is_file():
    sample_img = io.imread(sample_img_path); img_shape = sample_img.shape
    if img_shape[2] == 4:  # for png file, transfer 'a' channel for the image to RGB 3 channels image
        sample_img = rgba2rgb(sample_img); img_shape = sample_img.shape
        sample_img = img_as_ubyte(sample_img)
    # plt.figure(); plt.imshow(sample_img); plt.axis('off'); plt.tight_layout()
    # Unpack channels by index
    sample_img_red = sample_img[:, :, 0]; sample_img_green = sample_img[:, :, 1]; sample_img_blue = sample_img[:, :, 2]
    # Plotting all images on the same figure
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14.5, 6.5))
    axes[0, 0].imshow(sample_img); axes[0, 0].axis('off'); axes[0, 0].set_title("Original RGB Image")
    axes[0, 1].imshow(sample_img_red, cmap="Reds"); axes[0, 1].axis('off'); axes[0, 1].set_title("Red Channel")
    axes[0, 2].imshow(sample_img_green, cmap="Greens"); axes[0, 2].axis('off'); axes[0, 2].set_title("Green Channel")
    axes[0, 3].imshow(sample_img_blue, cmap="Blues"); axes[0, 3].axis('off'); axes[0, 3].set_title("Blue Channel")

    # Plotting histograms for channels
    hist_grey = histogram(sample_img_gray); norm_counts = np.asarray(hist_grey[0])/np.max(hist_grey[0])
    axes[1, 0].plot(hist_grey[1], norm_counts); axes[1, 0].set_title("Histogram Gray-Scaled Image")
    hist_r = histogram(sample_img_red); norm_counts_r = np.asarray(hist_r[0])/np.max(hist_r[0])
    axes[1, 1].plot(hist_r[1], norm_counts_r); axes[1, 1].set_title("Histogram Red Channel")
    hist_g = histogram(sample_img_green); norm_counts_g = np.asarray(hist_g[0])/np.max(hist_g[0])
    axes[1, 2].plot(hist_g[1], norm_counts_g); axes[1, 2].set_title("Histogram Green Channel")
    hist_b = histogram(sample_img_blue); norm_counts_b = np.asarray(hist_b[0])/np.max(hist_b[0])
    axes[1, 3].plot(hist_b[1], norm_counts_b); axes[1, 3].set_title("Histogram Blue Channel")
    plt.tight_layout()
