# -*- coding: utf-8 -*-
"""
Modelling of an acquired image distortion.

@author: sklykov
@license: The Unlicense
"""

# %% Imports
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.filters.rank import mean, median
from skimage.util import img_as_uint
from scipy.fft import fft2, fftshift
from matplotlib.colors import LogNorm
import time

# %% Parameters
k1 = 0.125; k2 = 0.0; k3 = 0.0  # for modelling distortion of an image
w = 250; h = 250  # default image sizes
grid_step = 5  # for generation of grid with points (one pixel)
plt.close("all")
# Flags for showing / calculation of variables
show_grid = False  # for showing using Matplotlib grid of points
show_fringes = True  # for showing using Matplotlib modelled interferometric fringes
compare_raw_vect_distortions = False  # for testing vectorization speed up of distortion calculation
test_vectorization = False  # for testing the vectorization speed up of distortion calculation
test_mean_filter = False; test_median_filter = False; test_mix_mean_median = False
test_filters = test_mean_filter or test_median_filter or test_mix_mean_median  # to enter filters testing
test_cropping = False  # test applying 1 (False) or 2 (True) methods below
print_test_information = True  # print out various testing / benchmarking information
if not compare_raw_vect_distortions and test_vectorization:
    w = 520; h = 560  # enlarged image sizes for performance checking
i_center = w // 2; j_center = h // 2  # generated image size properties
radius = np.min([i_center, j_center])  # define the radius of an image as the smallest geometrical size
calculate_filled_distorted_img = False
fringe_step_vert = 10  # defines the horizontal fringes density and width

# %% Grid generation
if show_grid:
    img = np.zeros((w, h), dtype='uint8')
    for i in range(grid_step, w, grid_step):
        for j in range(grid_step, h, grid_step):
            img[i, j] = 255
    # plt.figure(); plt.imshow(img)

# %% Distortion modelling
if show_grid:
    distorted_img = np.zeros((w, h), dtype='uint8')
    reconstructed_img = np.zeros((w, h), dtype='uint8')
    for i in range(grid_step, w, grid_step):
        for j in range(grid_step, h, grid_step):
            r = euclidean([i, j], [i_center, j_center])/radius  # normalized distance

            # Similar to the model of the relation of undistorted pixel coordinates to the distorted one
            i_distorted = i_center + int(np.round((i-i_center)*(1 + k1*r*r + k2*r*r*r*r + k3*np.power(r, 6)), 0))
            j_distorted = j_center + int(np.round((j-j_center)*(1 + k1*r*r + k2*r*r*r*r + k3*np.power(r, 6)), 0))
            # Reconstruction similar to the above one

            i_rec = i_distorted + int(np.round((i_distorted - i_center)*(1 + k1*r*r + k2*r*r*r*r + k3*np.power(r, 6)), 0))
            j_rec = j_distorted + int(np.round((j_distorted - j_center)*(1 + k1*r*r + k2*r*r*r*r + k3*np.power(r, 6)), 0))
            # Round mistake correction - ??? (reconstructed image violates grid pattern for one step)

            # Make images as grid of points
            if i_distorted > 0 and i_distorted < w and j_distorted > 0 and j_distorted < h:
                distorted_img[i_distorted, j_distorted] = 255
            if i_rec > 0 and i_rec < w and j_rec > 0 and j_rec < h:
                reconstructed_img[i_rec, j_rec] = 255
            # distorted_img[i, j] = 128

    # Show images
    plt.figure(); plt.imshow(distorted_img)
    plt.figure(); plt.imshow(reconstructed_img)

# %% Modelling interferometric fringes
img_fringes = np.zeros((w, h), dtype='float')
for i_fringe_center in range(fringe_step_vert//4, w, fringe_step_vert):
    # img_fringes[i-fringe_step_vert//4:i+fringe_step_vert//4, :] = 1.0  # straight, not blurred fringes
    # below - modelling the fringes as gaussian blurred fringes
    sigma_fringe = fringe_step_vert // 4
    i = i_fringe_center - fringe_step_vert // 2
    if i < 0:
        i = 0
    elif i >= w:
        i = w-1
    while i < i_fringe_center + (fringe_step_vert // 2):
        if i < w:
            fringe_profile = np.exp(-np.power(i-i_fringe_center, 2)/np.power(sigma_fringe, 2))
            img_fringes[i, :] = fringe_profile
            i += 1
        else:
            break

# Distortion of fringes
distorted_fringes_raw = np.zeros((w, h), dtype='float'); norm_radius = 1.0/radius
for i in range(w):
    for j in range(h):
        r = np.round(norm_radius*euclidean([i, j], [i_center, j_center]), 6)  # normalized distance
        # Similar to the model of the relation of undistorted pixel coordinates to the distorted one
        i_distorted = i_center + int(np.round((i-i_center)*(1 + k1*r*r), 0))
        j_distorted = j_center + int(np.round((j-j_center)*(1 + k1*r*r), 0))

        # Reassignment pixel values
        if i_distorted > 0 and i_distorted < w and j_distorted > 0 and j_distorted < h:
            distorted_fringes_raw[i_distorted, j_distorted] = img_fringes[i, j]

# Other distortion model
distorted_fringes_raw2 = np.zeros((w, h), dtype='float'); l1 = 1E-3
jj_u, ii_u = np.meshgrid(np.arange(start=0, stop=h, step=1), np.arange(start=0, stop=w, step=1))
radii_u = norm_radius*(np.power((ii_u-i_center), 2) + np.power((jj_u-j_center), 2))
radii_u_coeff = 1.0/(1.0 + l1*radii_u)
jj_d = j_center + np.int16(np.round((jj_u - j_center)*radii_u_coeff, 0))
ii_d = i_center + np.int16(np.round((ii_u - i_center)*radii_u_coeff, 0))
distorted_fringes_raw2[ii_d, jj_d] = img_fringes
# plt.figure(); plt.imshow(distorted_fringes_raw2)

# Restore artifacts introduced by the distortion - horizontally filling the gaps.
# Seems that symmetry of application - because of distorted horizontally aligned fringes
if calculate_filled_distorted_img:
    distorted_fringes_fill_hor = np.zeros((w, h), dtype='float')
    distorted_fringes_fill_hor += distorted_fringes_raw
    for i in range(w):
        for j in range(0, h-1, 1):
            if ((distorted_fringes_fill_hor[i, j] == 0.0 or distorted_fringes_fill_hor[i, j] < 1E-6)
               and distorted_fringes_fill_hor[i, j+1] > 0.0):
                distorted_fringes_fill_hor[i, j] = distorted_fringes_fill_hor[i, j+1]
        for j in range(1, h, 1):
            if ((distorted_fringes_fill_hor[i, j] == 0.0 or distorted_fringes_fill_hor[i, j] < 1E-6)
               and distorted_fringes_fill_hor[i, j-1] > 0.0):
                distorted_fringes_fill_hor[i, j] = distorted_fringes_fill_hor[i, j-1]
        for j in range(h-2, 0, -1):
            if ((distorted_fringes_fill_hor[i, j] == 0.0 or distorted_fringes_fill_hor[i, j] < 1E-6)
               and distorted_fringes_fill_hor[i, j+1] > 0.0):
                distorted_fringes_fill_hor[i, j] = distorted_fringes_fill_hor[i, j+1]
        for j in range(0, h-1, 1):
            if ((distorted_fringes_fill_hor[i, j] == 0.0 or distorted_fringes_fill_hor[i, j] < 1E-6)
               and distorted_fringes_fill_hor[i, j+1] > 0.0):
                distorted_fringes_fill_hor[i, j] = distorted_fringes_fill_hor[i, j+1]

# Interpolation of not assigned pixels using some interpolation (below - using 3x3 averaging mask) - Remove artifacts
distorted_fringes_interpol = np.zeros((w, h), dtype='float')
distorted_fringes_interpol += distorted_fringes_raw
for i in range(w):
    for j in range(h):
        if distorted_fringes_interpol[i, j] == 0.0 or distorted_fringes_interpol[i, j] < 1E-6:
            interpol_value = 0.0; summed_pixels = 0  # 3x3 mask interpolation - better than the 5x5 mask!
            for i_neighbour in range(i-1, i+2):
                for j_neighbour in range(j-1, j+2):
                    if i_neighbour >= 0 and i_neighbour < w and j_neighbour >= 0 and j_neighbour < h:
                        if distorted_fringes_interpol[i_neighbour, j_neighbour] > 0.0:
                            interpol_value += distorted_fringes_interpol[i_neighbour, j_neighbour]; summed_pixels += 1
            if summed_pixels > 0:
                distorted_fringes_interpol[i, j] = interpol_value / summed_pixels

# Additional smooth of interpolated values
distorted_fringes_interpol_sm = np.copy(distorted_fringes_interpol)
distorted_fringes_interpol_sm = gaussian(distorted_fringes_interpol_sm, sigma=0.85)

# Show images with fringes
if show_fringes:
    # plt.figure(); plt.imshow(distorted_fringes_raw)
    # plt.figure(); plt.imshow(distorted_fringes_interpol - distorted_fringes_raw)
    # plt.figure(); plt.imshow(distorted_fringes_interpol)
    plt.figure(); plt.imshow(distorted_fringes_interpol_sm)
    # Get Fourier transform of the distorted image
    fourier_img_draw = np.abs(fftshift(fft2(distorted_fringes_interpol_sm)))
    minF = np.min(fourier_img_draw); maxF = np.max(fourier_img_draw)
    plt.figure(); plt.imshow(fourier_img_draw, norm=LogNorm(vmin=minF, vmax=maxF))

# %% Back transformation of the distorted image and comparison of Fourier spectrums
restored_fringes_raw = np.zeros((w, h), dtype='float'); k1 = -k1+0.05

# Straight and slow implementation
if compare_raw_vect_distortions:
    t1 = time.time()
    for i in range(w):
        for j in range(h):
            r = np.round(euclidean([i, j], [i_center, j_center])/radius, 6)  # normalized distance
            # Similar to the model of the relation of undistorted pixel coordinates to the distorted one
            i_distorted = i_center + int(np.round((i-i_center)*(1 + k1*r*r), 0))
            j_distorted = j_center + int(np.round((j-j_center)*(1 + k1*r*r), 0))
            # Reassignment pixel values
            if i_distorted >= 0 and i_distorted < w and j_distorted >= 0 and j_distorted < h:
                restored_fringes_raw[i_distorted, j_distorted] = distorted_fringes_raw[i, j]
    print("Raw distortion takes ms:", int(round(1000*(time.time() - t1), 0)))

# Vectorization of the implementation above for speeding up the calculations
t1 = time.time()
i_coord_sq = np.arange(start=0, stop=w, step=1); j_coord_sq = np.arange(start=0, stop=h, step=1)
jj, ii = np.meshgrid(j_coord_sq, i_coord_sq)
i_coord_sq = np.power(i_coord_sq - i_center, 2); j_coord_sq = np.power(j_coord_sq - j_center, 2)
jj_coord_sq, ii_coord_sq = np.meshgrid(j_coord_sq, i_coord_sq)
radii_mesh_sq = np.power(np.sqrt(jj_coord_sq + ii_coord_sq)*norm_radius, 2)
restored_fringes_raw_vect = np.zeros((w, h), dtype='float')

# Calculation in for loop still performed
# for i in range(w):
#     for j in range(h):
#         # Similar to the model of the relation of undistorted pixel coordinates to the distorted one
#         i_distorted = i_center + int(np.round((i-i_center)*(1 + k1*radii_mesh_sq[i, j]), 0))
#         if i_distorted >= 0 and i_distorted < w:
#             j_distorted = j_center + int(np.round((j-j_center)*(1 + k1*radii_mesh_sq[i, j]), 0))
#             # Reassignment pixel values - transfer undistorted image to distorted one
#             if j_distorted >= 0 and j_distorted < h:
#                 restored_fringes_raw_vect[i_distorted, j_distorted] = distorted_fringes_raw[i, j]

# Moved out calculation from for loops, remaining only sorting out the coordinates
ii_dist = i_center + np.int16(np.round(((1.0 + k1*radii_mesh_sq)*(ii - i_center)), 0))
jj_dist = j_center + np.int16(np.round(((1.0 + k1*radii_mesh_sq)*(jj - j_center)), 0))

# i_distorted, j_distorted = 0, 0
# for i in range(w):
#     if np.min(jj_dist[i, :]) < h and np.min(ii_dist[i, :]) < w:
#         for j in range(h):
#             i_distorted = ii_dist[i, j]; j_distorted = jj_dist[i, j]
#             if j_distorted < h and i_distorted < w:
#                 restored_fringes_raw_vect[i_distorted, j_distorted] = distorted_fringes_raw[i, j]

# I've just noticed that the checks in for loops are not necessary to perform, indices lay in suitable regions
# So, it turns out that the simple loops are enough
# for i in range(w):
#     for j in range(h):
#         restored_fringes_raw_vect[ii_dist[i, j], jj_dist[i, j]] = distorted_fringes_raw[i, j]

# Or, even, going only on one row for coordinates conversion
# if w >= h:
#     for i in range(w):
#         restored_fringes_raw_vect[ii_dist[i, :], jj_dist[i, :]] = distorted_fringes_raw[i, :]
# else:
#     for j in range(h):
#         restored_fringes_raw_vect[ii_dist[:, j], jj_dist[:, j]] = distorted_fringes_raw[:, j]

# Test the direct assignment of matricies - to exclude the looping
restored_fringes_raw_vect[ii_dist, jj_dist] = distorted_fringes_raw  # ??? which image to distort for checking all algorithms
# restored_fringes_raw_vect[ii_dist, jj_dist] = img_fringes

# Check which pixels not included / used twice during transform
addressed_again_pixels = []; restored_fringes_raw_vect2 = np.zeros((w, h), dtype='float')
for i in range(w):
    for j in range(h):
        if restored_fringes_raw_vect2[ii_dist[i, j], jj_dist[i, j]] < 1E-9:
            restored_fringes_raw_vect2[ii_dist[i, j], jj_dist[i, j]] = img_fringes[i, j]
        else:
            addressed_again_pixels.append([ii_dist[i, j], jj_dist[i, j]])
            # restored_fringes_raw_vect2[ii_dist[i, j], jj_dist[i, j]] += img_fringes[i, j]
            # restored_fringes_raw_vect2[ii_dist[i, j], jj_dist[i, j]] *= 0.5
            restored_fringes_raw_vect2[ii_dist[i, j], jj_dist[i, j]] = 0.0
plt.figure(); plt.imshow(restored_fringes_raw_vect2)

if print_test_information:
    print("Vectorized distortion takes ms:", int(round(1000*(time.time() - t1), 0)))
if compare_raw_vect_distortions:
    plt.figure(); plt.imshow(restored_fringes_raw - restored_fringes_raw_vect)
restored_fringes_raw = restored_fringes_raw_vect  # use further the vectorized form

if test_cropping:
    # Crop the shrinked borders - all zero values in row or column
    i_crop_top = 0; j_crop_left = 0; i_crop_bottom = 0; j_crop_right = 0
    for i in range(w):
        if np.max(restored_fringes_raw[i, :]) > 0.0:
            i_crop_top = i; break
    for j in range(h):
        if np.max(restored_fringes_raw[:, j]) > 0.0:
            j_crop_left = j; break
    for i in range(i_crop_top+1, w):
        if np.max(restored_fringes_raw[i, :]) == 0.0:
            i_crop_bottom = i; break
    for j in range(j_crop_left+1, h):
        if np.max(restored_fringes_raw[:, j]) == 0.0:
            j_crop_right = j; break
    restored_fringes_raw = restored_fringes_raw[i_crop_top:i_crop_bottom, j_crop_left:j_crop_right]
w_restored, h_restored = restored_fringes_raw.shape

# Remove corners with zero pixels - conservative strategy to find first non-zero pixel on the diagonal
i_diag_top = 0; j_diag_left = 0; i_diag_bottom = 0; j_diag_right = 0
i, j = 0, 0
while not restored_fringes_raw[i, j] > 0.0:
    i += 1; j += 1
i_diag_top, j_diag_left = i, j
i, j = w_restored-1, h_restored-1
while not restored_fringes_raw[i, j] > 0.0:
    i -= 1; j -= 1
i_diag_bottom, j_diag_right = i, j
restored_fringes_raw = restored_fringes_raw[i_diag_top:i_diag_bottom, j_diag_left:j_diag_right]
w_restored, h_restored = restored_fringes_raw.shape

if test_cropping and print_test_information:
    print("Cropped image sizes after 2 methods:", w_restored, h_restored)
else:
    print("Cropped image sizes after 1 method:", w_restored, h_restored)

# Interpolation of not assigned pixels using some interpolation (below - using 3x3 averaging mask) - Remove artifacts
t1 = time.time(); restored_fringes_interpol = np.copy(restored_fringes_raw)
interpolation_sum_coeffs = np.asarray([1.0, 0.5, 1/3, 0.25, 0.2, 1/6, 1/7, 0.125])
# restored_fringes_interpol2 = np.copy(restored_fringes_interpol)

# Straight loop over the image
# for i in range(1, w_restored-1):
#     for j in range(1, h_restored-1):
#         # For excluding borders and all special (edges) cases, above - counting start and ends in border sizes +/- 1
#         if restored_fringes_interpol2[i, j] < 1E-6:

#             # Manual interpolation calculation
#             # interpol_value = 0.0; summed_pixels = 0  # 3x3 mask interpolation - better than the 5x5 mask!
#             # for i_neighbour in range(i-1, i+2):
#             #     for j_neighbour in range(j-1, j+2):
#             #         if i_neighbour >= 0 and i_neighbour < w_restored and j_neighbour >= 0 and j_neighbour < h_restored:
#             #             if restored_fringes_interpol[i_neighbour, j_neighbour] > 0.0:
#             #                 interpol_value += restored_fringes_interpol[i_neighbour, j_neighbour]; summed_pixels += 1
#             # if summed_pixels > 0:
#             #     restored_fringes_interpol[i, j] = interpol_value / summed_pixels

#             # Define mask coordinates - for calculation interpolation sum
#             i_top = i-1; i_bottom = i+1; j_left = j-1; j_right = j+1
#             # Calculate sum of pixels inside 3x3 mask using the numpy
#             interpol_mask = restored_fringes_interpol2[i_top:i_bottom+1, j_left:j_right+1]
#             restored_fringes_interpol2[i, j] = np.sum(interpol_mask)*interpolation_sum_coeff
# restored_fringes_interpol2 = restored_fringes_interpol2[1:w_restored-1, 1:h_restored-1]

# Speed up by looping only on the calculated pixels of interest - vacant pixels not filled after conversion
restored_fringes_interpol_wt_borders = restored_fringes_interpol[1:w_restored-1, 1:h_restored-1]
plt.figure(); plt.imshow(restored_fringes_interpol_wt_borders)
zero_ii, zero_jj = np.nonzero(restored_fringes_interpol_wt_borders < 1E-6)  # controvercially, returns indices of zero pixels
for zero_index in range(zero_ii.shape[0]):
    i, j = zero_ii[zero_index] + 1, zero_jj[zero_index] + 1
    # Define mask coordinates - for calculation interpolation sum
    i_top = i-1; i_bottom = i+1; j_left = j-1; j_right = j+1
    # Calculate sum of pixels inside 3x3 mask using the numpy
    zero_mask_ii, _ = np.nonzero(restored_fringes_interpol[i_top:i_bottom+1, j_left:j_right+1] > 1E-6)
    restored_fringes_interpol_wt_borders[i-1, j-1] = (interpolation_sum_coeffs[zero_mask_ii.shape[0]-1]
                                                      * np.sum(restored_fringes_interpol[i_top:i_bottom+1, j_left:j_right+1]))
if print_test_information:
    print("Interpolation takes ms:", int(round(1000*(time.time() - t1), 0)))

# Avoiding interpolation of only selected pixels by applying some filter and removing artefacts
if test_filters:
    restored_fringes_filter = np.copy(restored_fringes_raw)
    restored_fringes_filter = img_as_uint(restored_fringes_filter)  # need for mean, median filters
    footprint = np.asarray([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    if test_mean_filter:
        for iteration in range(2):
            restored_fringes_filter = mean(restored_fringes_filter, footprint)
        plt.figure(); plt.imshow(restored_fringes_filter)
    if test_median_filter and not test_mean_filter and not test_mix_mean_median:
        for iteration in range(2):
            restored_fringes_filter = median(restored_fringes_filter, footprint)
        plt.figure(); plt.imshow(restored_fringes_filter)
    if test_mix_mean_median and not test_median_filter and not test_mean_filter:
        restored_fringes_filter = median(restored_fringes_filter, footprint)
        restored_fringes_filter = median(restored_fringes_filter, footprint)
        restored_fringes_filter = mean(restored_fringes_filter, footprint)
        plt.figure(); plt.imshow(restored_fringes_filter)
    # restored_fringes_filter = median(restored_fringes_filter)  # Tested, not closing all artifacts

# Additional smooth of interpolated values
# t1 = time.time()
restored_fringes_interpol_sm = np.copy(restored_fringes_interpol_wt_borders)
restored_fringes_interpol_sm = gaussian(restored_fringes_interpol_sm, sigma=0.85)
# print("Gaussian smoothing takes ms:", int(round(1000*(time.time() - t1), 0)))

# Plotting rhe results of reconstruction
if show_fringes:
    # plt.figure(); plt.imshow(restored_fringes_raw)
    # plt.figure(); plt.imshow(restored_fringes_interpol)
    plt.figure(); plt.imshow(restored_fringes_interpol_sm)

    # Get Fourier transform of the restored image
    fourier_rest_img_draw = np.abs(fftshift(fft2(restored_fringes_interpol_sm)))
    minF = np.min(fourier_rest_img_draw); maxF = np.max(fourier_rest_img_draw)
    plt.figure(); plt.imshow(fourier_rest_img_draw, norm=LogNorm(vmin=minF, vmax=maxF))

# %% Testing
if __name__ == "__main__":
    pass
