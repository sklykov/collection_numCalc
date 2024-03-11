# -*- coding: utf-8 -*-
"""
Tests of DCT calculation.

@author: sklykov
@license: The Unlicense

"""
# %% Imports
# import numpy as np
from scipy.fftpack import dct
from LoaderFile import loadSampleImage  # loading sample from 'resources' folder
import matplotlib.pyplot as plt
import numpy as np

# %% Conversion physical units to the resolved frequencies
wavelength_um = 0.532; NA = 0.95
res = 0.61*wavelength_um/NA; nyq_pix_size = round(0.5*res, 9)
k_max = round(2.0*np.pi/res, 6)  # max resolved spatial frequency

# about the frequency definition for Fourier transform: https://en.wikipedia.org/wiki/Fourier_transform#Angular_frequency_(%CF%89)
# implementation details of numpy FFT (hint about frequency def.): https://numpy.org/doc/stable/reference/routines.fft.html#implementation-details

# %% Tests run as the main script
if __name__ == "__main__":
    plt.close("all")
    sample_img = loadSampleImage(); h, w = sample_img.shape; ratio = h/w; w_fig = 7.2
    img_dct = dct(sample_img); img_dct_log = np.log(np.abs(img_dct))
    plt.figure("Sample", figsize=(w_fig, w_fig*ratio)); plt.imshow(sample_img, cmap=plt.cm.gray); plt.tight_layout()
    # the raw DCT result isn't useful for representation of coefficient containing image information, use log instead
    # reference: https://mathworks.com/help/images/ref/dct2.html
    # plt.figure("Raw DCT", figsize=(w_fig, w_fig*ratio)); plt.imshow(img_dct, cmap=plt.cm.viridis); plt.tight_layout()
    plt.figure("Log DCT", figsize=(w_fig, w_fig*ratio)); plt.imshow(img_dct_log, cmap=plt.cm.magma); plt.tight_layout()
