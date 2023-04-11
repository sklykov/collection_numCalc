# -*- coding: utf-8 -*-
"""
Experiments with creation of attractors / repulsers - something resembling the electrostatic field.

@author: sklykov
@license: The Unlicense
"""

# %% Imports
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt


# %% Parameters
k1 = 0.1; k2 = 0.0; k3 = 0.0
w = 100; h = 100; grid_step = 5
plt.close("all")


# %% Grid generation
img = np.zeros((w, h), dtype='uint8')
for i in range(grid_step, w, grid_step):
    for j in range(grid_step, h, grid_step):
        img[i, j] = 255
# plt.figure(); plt.imshow(img)

# %% Distortion modelling
i_center = w // 2; j_center = h // 2
radius = np.min([i_center, j_center])
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

# %% Testing
if __name__ == "__main__":
    pass
