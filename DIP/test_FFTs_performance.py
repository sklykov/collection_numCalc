# -*- coding: utf-8 -*-
"""
Measuring performance of various FFT functions applied on an image.

@author: sklykov
@license: The Unlicense

"""

# %% Global imports
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as npfft
from scipy.fft import fft2, fftshift, rfft2, ifft2
from matplotlib.colors import LogNorm
from skimage.filters import gaussian
import time
import random
import os

# For using and testing numba package, note its limitations on Python and Numpy version!
numba_installed = False
try:
    from numba import jit
    numba_installed = True
except ModuleNotFoundError:
    print("The package numba isn't installed, functions with jit compilation won't be tested")


# %% Testing parameters
compare_scipy_numpy = False

# %% Sample image generation
plt.close('all')
w = 2560; h = 2480  # default image sizes - high resolution image
img_fringes = np.zeros((h, w), dtype='uint16')  # 16bit image
# Modelling of image as the set of interferometric fringes
fringe_step_hor = 42  # defines the horizontal fringes density and width
fringe_width = 32  # in pixels
max_fringe_val = (2**16 - 1)//2  # max pixel value on the center
y_fringe_center = fringe_step_hor // 5

# Generate horizontal fringes
while y_fringe_center < h:
    # below - modelling the fringes as gaussian blurred fringes
    sigma_fringe = fringe_width / 4
    y = y_fringe_center - fringe_step_hor // 2
    if y < 0:
        y = 0
    elif y >= h:
        y = h-1
    while y < y_fringe_center + (fringe_step_hor // 2):
        if y < h:
            fringe_profile = np.uint16(np.round((max_fringe_val
                                       * np.exp(-np.power(y-y_fringe_center, 2)/np.power(sigma_fringe, 2))), 0))
            img_fringes[y, :] = fringe_profile
            y += 1
        else:
            break
    # Change the period between fringes randomly
    y_fringe_center += random.choice([fringe_step_hor - 4, fringe_step_hor - 2, fringe_step_hor,
                                      fringe_step_hor + 1, fringe_step_hor + 3, fringe_step_hor + 5])

# For getting distributed Fourier spectrum - bend the fringes by applying modelled barrel distortion
i_center = w // 2; j_center = h // 2; norm_radius = 1.0 / (np.min([w, h])); k1 = -0.12
i_coord_sq = np.arange(start=0, stop=w, step=1); j_coord_sq = np.arange(start=0, stop=h, step=1)
ii, jj = np.meshgrid(i_coord_sq, j_coord_sq)
i_coord_sq = np.power(i_coord_sq - i_center, 2); j_coord_sq = np.power(j_coord_sq - j_center, 2)
ii_coord_sq, jj_coord_sq = np.meshgrid(i_coord_sq, j_coord_sq)
radii_mesh_sq = np.power(np.sqrt(jj_coord_sq + ii_coord_sq)*norm_radius, 2)
bended_fringes = np.zeros((h, w), dtype='float')
ii_dist = i_center + np.int16(np.round(((1.0 + k1*radii_mesh_sq)*(ii - i_center)), 0))
jj_dist = j_center + np.int16(np.round(((1.0 + k1*radii_mesh_sq)*(jj - j_center)), 0))
bended_fringes[jj_dist, ii_dist] = img_fringes
i_top = np.min(ii_dist[0, :]); i_bottom = np.max(ii_dist[h-1, :])
j_left = np.min(jj_dist[:, 0]); j_right = np.max(jj_dist[:, w-1])
bended_fringes = bended_fringes[j_left:j_right, i_top:i_bottom]
bended_fringes = gaussian(bended_fringes, sigma=0.95)

# Visualize generation results + adding evenly distributed random noise
# img_fringes += np.random.randint(max_fringe_val//20, high=max_fringe_val//5, size=(h, w), dtype='uint16')
# plt.figure(); plt.imshow(img_fringes); plt.tight_layout()  # draw generated image - only vertical lines
plt.figure(); plt.imshow(bended_fringes); plt.tight_layout()
h_distorted, w_distorted = bended_fringes.shape


# %% Generate fringes with noise
def generate_noisy_fringes():
    noisy_fringes = np.copy(bended_fringes)
    noisy_fringes += np.random.randint(max_fringe_val//25, high=max_fringe_val//5,
                                       size=(h_distorted, w_distorted), dtype='uint16')
    return noisy_fringes


# %% Testing functions - wrappers for providing same interface for function calls
def scipy_fft2(img):
    return np.abs(fftshift(fft2(img)))


def pure_numpy_fft2(img):
    return np.abs(npfft.fftshift(npfft.fft2(img)))


if numba_installed:
    # default compilation with option nopython=True produced the error and forces to use
    # below parameter, but it even makes computation slower
    # @njit also produces errors
    @jit(forceobj=True)
    def jit_numpy_fft2(img: np.ndarray):
        return np.abs(npfft.fftshift(npfft.fft2(img)))


# %% Measuring performance
t_mean_ms = 0; n_iterations = 5

# Generate sample images
img_array = [generate_noisy_fringes() for i in range(n_iterations)]

# Measure performance - Scipy, reference implementation
for i in range(n_iterations):
    t1 = time.time()
    scipy_fft2(img_array[i])
    t2 = time.time()
    t_mean_ms += 1000.0*(t2-t1)
t_mean_ms = int(np.round(t_mean_ms / n_iterations, 0))
print(f"Scipy fft2 + fftshift + np.abs ({w_distorted}x{h_distorted} image) ms:", t_mean_ms)


# Measure performance - Scipy, reference implementation
avalaible_threads = os.cpu_count()
print("Detected workers (threads):", avalaible_threads)
if avalaible_threads//2 - 1 > 2:
    set_n_workers = avalaible_threads//2
else:
    set_n_workers = 2
for i in range(n_iterations):
    t1 = time.time()
    np.abs(fftshift(fft2(img_array[i], workers=set_n_workers)))
    t2 = time.time()
    t_mean_ms += 1000.0*(t2-t1)
t_mean_ms = int(np.round(t_mean_ms / n_iterations, 0))
print(f"Scipy fft2 + set {set_n_workers} workers ({w_distorted}x{h_distorted} image) ms:", t_mean_ms)


# Measure performance - Numpy
if compare_scipy_numpy:
    for i in range(n_iterations):
        t1 = time.time()
        pure_numpy_fft2(img_array[i])
        t2 = time.time()
        t_mean_ms += 1000.0*(t2-t1)
    t_mean_ms = int(np.round(t_mean_ms / n_iterations, 0))
    print(f"Pure Numpy fft2 + fftshift + np.abs ({w_distorted}x{h_distorted} image) ms:", t_mean_ms)


# Measure performance - Numpy + numba jit accelaration
if numba_installed and compare_scipy_numpy:
    for i in range(n_iterations):
        t1 = time.time()
        jit_numpy_fft2(img_array[i])
        t2 = time.time()
        t_mean_ms += 1000.0*(t2-t1)
    t_mean_ms = int(np.round(t_mean_ms / n_iterations, 0))
    print(f"JIT compiled Numpy fft2 + fftshift + np.abs ({w_distorted}x{h_distorted} image) ms:", t_mean_ms)

# Visualize Fourier spectrum - only central part
fourier_img = np.abs(fftshift(fft2(img_array[0])))
hf, wf = fourier_img.shape
# fourier_img_draw = fourier_img[hf//2 - hf//6:hf//2+hf//6, wf//2 - wf//6: wf//2 + wf//6]
fourier_img_draw = fourier_img
minF = np.min(fourier_img_draw); maxF = np.max(fourier_img_draw)
plt.figure(); plt.imshow(fourier_img_draw, norm=LogNorm(vmin=minF, vmax=maxF)); plt.tight_layout()

# Alternative Fourier transform - real Fourier transform, output - only half of absolute Fourier spectrum
fourier_img_half = np.abs(fftshift(rfft2(img_array[0]))); fourier_img_half2 = np.copy(fourier_img_half)
hf_half, wf_half = fourier_img_half.shape
fourier_img_half_crop1 = fourier_img_half[:, ((wf_half-1)//2):wf_half:1]
fourier_img_half_crop2 = fourier_img_half2[:, 0:((wf_half-1)//2):1]  # only cropping out from copied array allows this operation
fourier_img_half[:, 0:(((wf_half-1)//2)+1):1] = fourier_img_half_crop1
fourier_img_half[:, (((wf_half-1)//2)+1):wf_half:1] = fourier_img_half_crop2
fourier_img_half_flipped = np.flip(fourier_img_half, axis=1)
fourier_img_conv = np.copy(fourier_img)
fourier_img_conv[:, 0:(((wf-1)//2)+1):1] = fourier_img_half_flipped
fourier_img_conv[:, ((wf-1)//2):wf:1] = fourier_img_half

# Plot the transforms for confirmation
# fourier_img_half_draw = fourier_img_conv
# minF = np.min(fourier_img_half_draw); maxF = np.max(fourier_img_half_draw)
# plt.figure(); plt.imshow(fourier_img_half_draw, norm=LogNorm(vmin=minF, vmax=maxF)); plt.tight_layout()


# Check how the initial image restored after using real Fourier transform and some manipulations
def real_fft2_conversion(img, workers):
    fourier_img_half_imaginery = rfft2(img, workers=workers); fourier_img_half_imaginery2 = np.copy(fourier_img_half_imaginery)
    fourier_img_half_crop1_imaginery = fourier_img_half_imaginery[:, ((wf_half-1)//2):wf_half:1]
    fourier_img_half_crop2_imaginery = fourier_img_half_imaginery2[:, 0:((wf_half-1)//2):1]
    fourier_img_half_imaginery[:, 0:(((wf_half-1)//2)+1):1] = fourier_img_half_crop1_imaginery
    fourier_img_half_imaginery[:, (((wf_half-1)//2)+1):wf_half:1] = fourier_img_half_crop2_imaginery
    fourier_img_half_flipped_imaginery = np.flip(fourier_img_half_imaginery, axis=1)
    fourier_img_conv_imaginery = np.zeros((hf, wf), dtype="complex128")
    fourier_img_conv_imaginery[:, 0:(((wf-1)//2)+1):1] = fourier_img_half_flipped_imaginery
    fourier_img_conv_imaginery[:, ((wf-1)//2):wf:1] = fourier_img_half_imaginery
    return fourier_img_conv_imaginery


# Plot for checking visually the restored image after spectrum recreation
# restored_img = np.abs(ifft2(real_fft2_conversion(img_array[0], workers=set_n_workers)))
# plt.figure(); plt.imshow(restored_img); plt.tight_layout()

# Composing the transformations above for checking its performance
for i in range(n_iterations):
    t1 = time.time()
    np.abs(ifft2(real_fft2_conversion(img_array[i], workers=set_n_workers)))
    t2 = time.time()
    t_mean_ms += 1000.0*(t2-t1)
t_mean_ms = int(np.round(t_mean_ms / n_iterations, 0))
print(f"Scipy rfft2 + conversion + {set_n_workers} workers ({w_distorted}x{h_distorted} image) ms:", t_mean_ms)
