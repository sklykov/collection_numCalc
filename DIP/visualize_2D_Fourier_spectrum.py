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
    w_f = 120; h_f = 84  # will be converted to the image sizes

    # Demonstration of on axis frequencies comversion to an image
    centralized_spectrum = np.zeros(shape=(h_f, w_f))  # bare spectrum
    # Note that the overall image intensity, for getting max intensity on the recovered image ~ 1.0 => used multiplication of the image sizes
    # centralized_spectrum[h_f//2, w_f//2] = w_f*h_f  # central frequency specifying the intensity on the image
    centralized_spectrum[h_f//2, w_f//2] = 1.0  # normalized central frequency for better plotting other frequencies
    centralized_spectrum[h_f//2, w_f//2 + w_f//10] = 1.0  # side frequency, resolves to 120/10 = 12 vertical stripes presented on the figure below
    centralized_spectrum[h_f//2, w_f//2 - w_f//10] = 0.5  # symmetrical side frequency, but it adds also intensity between stripes
    plt.figure("Centralized Fourier spatial spectrum with on-axis frequencies"); plt.imshow(centralized_spectrum)
    figure = np.round(np.abs(ifft2(centralized_spectrum))*w_f*h_f, 6)  # scaled up the pixel intensities of an image - result of inverse FT
    plt.figure("Restored from Fourier spectrum (on-axis freq.) Image"); plt.imshow(figure)

    # Demonstration of combination of frequencies conversion to an image
    centralized_spectrum = np.zeros(shape=(h_f, w_f))  # bare spectrum
    centralized_spectrum[h_f//2, w_f//2] = 1.0  # normalized central frequency for better plotting other frequencies
    centralized_spectrum[h_f//2 - h_f//12, w_f//2 + w_f//15] = 0.8; centralized_spectrum[h_f//2 + h_f//16, w_f//2 - w_f//11] = 0.75
    centralized_spectrum[h_f//2 + h_f//10, w_f//2 + w_f//8] = 0.65
    plt.figure("Centralized Fourier spatial spectrum with freq. combination"); plt.imshow(centralized_spectrum)
    figure = np.round(np.abs(ifft2(centralized_spectrum))*w_f*h_f, 6)  # scaled up the pixel intensities of an image - result of inverse FT
    plt.figure("Restored from Fourier spectrum (freq. combination) Image"); plt.imshow(figure)

    # Demonstration of high frequencies conversion to an image
    centralized_spectrum = np.zeros(shape=(h_f, w_f))  # bare spectrum
    centralized_spectrum[h_f//2, w_f//2] = 1.0  # normalized central frequency for better plotting other frequencies
    centralized_spectrum[h_f//2, w_f//2 + int(round(w_f/2.25, 0))] = 0.75; centralized_spectrum[h_f//2 - int(round(h_f/2.25, 0)), w_f//2] = 0.75
    plt.figure("Centralized Fourier spatial spectrum with high frequencies"); plt.imshow(centralized_spectrum)
    figure = np.round(np.abs(ifft2(centralized_spectrum))*w_f*h_f, 6)  # scaled up the pixel intensities of an image - result of inverse FT
    plt.figure("Restored from Fourier spectrum (high freq.) Image"); plt.imshow(figure)
