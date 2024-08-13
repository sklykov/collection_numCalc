# -*- coding: utf-8 -*-
"""
Visualizing 2D FFT spectrum for the reference.

Inspired by: https://docs.scipy.org/doc/scipy/tutorial/fft.html#and-n-d-discrete-fourier-transforms

@author: sklykov
@license: The Unlicense

"""
# %% Global imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import ifft2

# %% Tests
if __name__ == "__main__":
    plt.close('all')
    w_f = 120; h_f = 80  # will be converted to the image sizes
    centralized_spectrum = np.zeros(shape=(h_f, w_f))
    centralized_spectrum[h_f//2, w_f//2] = w_f*h_f  # central frequency specifying the intensity on the image
    # Note that the overall image intensity, for getting max intensity on the recovered image ~ 1.0 => used multiplication of the image sizes
    centralized_spectrum[h_f//2, w_f//2 + w_f//10] = 2.5  # side frequency, resolves to 120/10 = 12 vertical stripes presented on the figure below
    plt.figure("Centralized Fourier spatial spectrum"); plt.imshow(centralized_spectrum)
    figure = np.round(np.abs(ifft2(centralized_spectrum)), 6)
    plt.figure("Restored from Fourier spectrum Figure"); plt.imshow(figure)
