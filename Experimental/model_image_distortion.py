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


# %% Parameters
k1 = 0.085; k2 = 0.0; k3 = 0.0
w = 400; h = 400; grid_step = 5
plt.close("all")
# Flags for showing / calculation of variables
show_grid = False  # for showing using Matplotlib grid of points
show_fringes = True  # for showing using Matplotlib modelled interferometric fringes
calculate_filled_distorted_img = False
fringe_step_vert = 14

# %% Grid generation
if show_grid:
    img = np.zeros((w, h), dtype='uint8')
    for i in range(grid_step, w, grid_step):
        for j in range(grid_step, h, grid_step):
            img[i, j] = 255
    # plt.figure(); plt.imshow(img)

# %% Distortion modelling
i_center = w // 2; j_center = h // 2
radius = np.min([i_center, j_center])
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
distorted_fringes_raw = np.zeros((w, h), dtype='float')
distorted_fringes_fill_hor = np.zeros((w, h), dtype='float')
for i in range(w):
    for j in range(h):
        r = euclidean([i, j], [i_center, j_center])/radius  # normalized distance
        # Similar to the model of the relation of undistorted pixel coordinates to the distorted one
        i_distorted = i_center + int(np.round((i-i_center)*(1 + k1*r*r), 0))
        j_distorted = j_center + int(np.round((j-j_center)*(1 + k1*r*r), 0))

        # Reassignment pixel values
        if i_distorted > 0 and i_distorted < w and j_distorted > 0 and j_distorted < h:
            distorted_fringes_raw[i_distorted, j_distorted] = img_fringes[i, j]

# Restore artifacts introduced by the distortion - horizontally filling the gaps.
# Seems that symmetry of application - because of distorted horizontally aligned fringes
if calculate_filled_distorted_img:
    distorted_fringes_fill_hor += distorted_fringes_raw
    for i in range(w):
        for j in range(0, h-1, 1):
            if (distorted_fringes_fill_hor[i, j] == 0.0 or distorted_fringes_fill_hor[i, j] < 1E-6) and distorted_fringes_fill_hor[i, j+1] > 0.0:
                distorted_fringes_fill_hor[i, j] = distorted_fringes_fill_hor[i, j+1]
        for j in range(1, h, 1):
            if (distorted_fringes_fill_hor[i, j] == 0.0 or distorted_fringes_fill_hor[i, j] < 1E-6) and distorted_fringes_fill_hor[i, j-1] > 0.0:
                distorted_fringes_fill_hor[i, j] = distorted_fringes_fill_hor[i, j-1]
        for j in range(h-2, 0, -1):
            if (distorted_fringes_fill_hor[i, j] == 0.0 or distorted_fringes_fill_hor[i, j] < 1E-6) and distorted_fringes_fill_hor[i, j+1] > 0.0:
                distorted_fringes_fill_hor[i, j] = distorted_fringes_fill_hor[i, j+1]
        for j in range(0, h-1, 1):
            if (distorted_fringes_fill_hor[i, j] == 0.0 or distorted_fringes_fill_hor[i, j] < 1E-6) and distorted_fringes_fill_hor[i, j+1] > 0.0:
                distorted_fringes_fill_hor[i, j] = distorted_fringes_fill_hor[i, j+1]

# Interpolation of not assigned pixels using some interpolation
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
distorted_fringes_interpol_sm = gaussian(distorted_fringes_interpol_sm, sigma=1.0)


# Show images with fringes
if show_fringes:
    # plt.figure(); plt.imshow(distorted_fringes_raw)
    # plt.figure(); plt.imshow(distorted_fringes_interpol - distorted_fringes_raw)
    plt.figure(); plt.imshow(distorted_fringes_interpol)
    plt.figure(); plt.imshow(distorted_fringes_interpol_sm)

# %% Testing
if __name__ == "__main__":
    pass
