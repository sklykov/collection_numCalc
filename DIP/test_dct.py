# -*- coding: utf-8 -*-
"""
Tests of DCT calculation.

@author: sklykov
@license: The Unlicense

"""
# %% Imports
# import numpy as np
from scipy.fftpack import dct, idct
from LoaderFile import loadSampleImage  # loading sample from 'resources' folder
import matplotlib.pyplot as plt
import numpy as np

# %% Conversion physical units to the resolved frequencies
wavelength_um = 0.532; NA = 0.95
resolution_um = round(0.61*wavelength_um/NA, 6)  # max resolution of an optical system (objective) in um
nyq_pixel_size = round(0.5*resolution_um, 9)  # Nyquist pixel size (represents a physical length of an object shown in a pixel)
pixel_size = nyq_pixel_size/10.0  # assuming that actual pixel size is little bit smaller than ultimate requirement - Nyquist size
# k_max = round(2.0*np.pi/res, 6)  # max resolved spatial frequency
spatial_frequency_unit = round(1.0/pixel_size, 9)  # the unit of the spatial frequency, like Hz and seconds - for DFT, DCT
length_resolution_pixels = round(resolution_um/pixel_size, 9)   # can be used for direct limiting of the spatial frequency resolved by the system

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

    # Replicate cut off unresolved frequencies and restore the image
    freq_h = int(round(length_resolution_pixels, 0) + 1); freq_w = int(round(length_resolution_pixels/ratio, 0) + 1)
    cropped_img_dct = np.zeros(shape=(h, w)); cropped_img_dct[:, 0:freq_w] = img_dct[:, 0:freq_w]  # !!! cropping only on columns, rows - repeated
    plt.figure("Cropped Log DCT Pixel Size Based", figsize=(w_fig, w_fig*ratio))
    plt.imshow(np.log(np.abs(cropped_img_dct) + 1.0), cmap=plt.cm.magma); plt.tight_layout()
    # restoring of the image based on the cropped DCT spectrum, scaling it back to 8 bit initial image
    restored_img_dct = np.abs(idct(cropped_img_dct)); restored_img_dct *= 255.0/np.max(restored_img_dct)
    restored_img_dct = np.round(restored_img_dct, 0).astype(np.uint8)
    plt.figure("Restored Pixel Size Based", figsize=(w_fig, w_fig*ratio)); plt.imshow(restored_img_dct, cmap=plt.cm.gray); plt.tight_layout()

    # Increase in ... times cropping frequency
    mf = 10; freq_w *= mf; cropped_img_dct = np.zeros(shape=(h, w)); cropped_img_dct[:, 0:freq_w] = img_dct[:, 0:freq_w]
    plt.figure(f"Cropped Log DCT {mf}*Size", figsize=(w_fig, w_fig*ratio))
    plt.imshow(np.log(np.abs(cropped_img_dct) + 1.0), cmap=plt.cm.magma); plt.tight_layout()
    # restoring of the image based on the cropped DCT spectrum, scaling it back to 8 bit initial image
    restored_img_dct = np.abs(idct(cropped_img_dct)); restored_img_dct *= 255.0/np.max(restored_img_dct)
    restored_img_dct = np.round(restored_img_dct, 0).astype(np.uint8)
    plt.figure(f"Restored Pixel {mf}*Size", figsize=(w_fig, w_fig*ratio)); plt.imshow(restored_img_dct, cmap=plt.cm.gray); plt.tight_layout()

    # Cropping only half of frequencies
    freq_w = w // 2; cropped_img_dct = np.zeros(shape=(h, w)); cropped_img_dct[:, 0:freq_w] = img_dct[:, 0:freq_w]
    plt.figure("Cropped Log DCT Half", figsize=(w_fig, w_fig*ratio))
    plt.imshow(np.log(np.abs(cropped_img_dct) + 1.0), cmap=plt.cm.magma); plt.tight_layout()
    # restoring of the image based on the cropped DCT spectrum, scaling it back to 8 bit initial image
    restored_img_dct = np.abs(idct(cropped_img_dct)); restored_img_dct *= 255.0/np.max(restored_img_dct)
    restored_img_dct = np.round(restored_img_dct, 0).astype(np.uint8)
    plt.figure("Restored Pixel Half", figsize=(w_fig, w_fig*ratio)); plt.imshow(restored_img_dct, cmap=plt.cm.gray); plt.tight_layout()
    plt.figure("Diff. Half Restored - Original", figsize=(w_fig, w_fig*ratio))
    plt.imshow(np.abs(sample_img - restored_img_dct), cmap=plt.cm.viridis); plt.tight_layout()

    # Direct Restore
    freq_w = w; cropped_img_dct = np.zeros(shape=(h, w)); cropped_img_dct[:, 0:freq_w] = img_dct[:, 0:freq_w]
    # restoring of the image based on the cropped DCT spectrum, scaling it back to 8 bit initial image
    restored_img_dct = np.abs(idct(cropped_img_dct)); restored_img_dct *= 255.0/np.max(restored_img_dct)
    restored_img_dct = np.round(restored_img_dct, 0).astype(np.uint8)
    plt.figure("Diff. Full Restored - Original", figsize=(w_fig, w_fig*ratio))
    plt.imshow(np.abs(sample_img - restored_img_dct), cmap=plt.cm.viridis); plt.tight_layout()

    # Partial crop and restore
    freq_w = w // 2; freq_h = h // 2; cropped_img_dct = np.zeros(shape=(h, w)); cropped_img_dct[0:freq_h, 0:freq_w] = img_dct[0:freq_h, 0:freq_w]
    plt.figure("Cropped Log DCT Half&Half", figsize=(w_fig, w_fig*ratio))
    plt.imshow(np.log(np.abs(cropped_img_dct) + 1.0), cmap=plt.cm.magma); plt.tight_layout()
    # restoring of the image based on the cropped DCT spectrum, scaling it back to 8 bit initial image
    restored_img_dct = np.abs(idct(cropped_img_dct)); restored_img_dct *= 255.0/np.max(restored_img_dct)
    restored_img_dct = np.round(restored_img_dct, 0).astype(np.uint8)
    plt.figure("Restored Pixel Half&Half", figsize=(w_fig, w_fig*ratio)); plt.imshow(restored_img_dct, cmap=plt.cm.gray); plt.tight_layout()

    # Circle cropping of frequencies
    r = min(w//2, h // 2); cropped_img_dct = np.zeros(shape=(h, w))
    for i in range(h):
        for j in range(w):
            dist1 = np.sqrt(i*i + j*j); dist2 = np.sqrt(np.power(h-i, 2) + np.power(j, 2))
            if dist1 <= r:
                cropped_img_dct[i, j] = img_dct[i, j]
            elif dist2 <= r:
                cropped_img_dct[i, j] = img_dct[i, j]
            else:
                break
    plt.figure("Cropped Log DCT Radius", figsize=(w_fig, w_fig*ratio))
    plt.imshow(np.log(np.abs(cropped_img_dct) + 1.0), cmap=plt.cm.magma); plt.tight_layout()
    # restoring of the image based on the cropped DCT spectrum, scaling it back to 8 bit initial image
    restored_img_dct = np.abs(idct(cropped_img_dct)); restored_img_dct *= 255.0/np.max(restored_img_dct)
    restored_img_dct = np.round(restored_img_dct, 0).astype(np.uint8)
    plt.figure("Restored Pixel Radius", figsize=(w_fig, w_fig*ratio)); plt.imshow(restored_img_dct, cmap=plt.cm.gray); plt.tight_layout()
