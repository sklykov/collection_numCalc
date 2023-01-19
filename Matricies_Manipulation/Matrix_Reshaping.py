"""
Matricies editing - reshaping 1D matrix to the 2D with special columns.

@author: sklykov
@license: The Unlicense
"""

# %% Imports
import numpy as np

# %% Demo
coordinates = [100, 200, 50, 400]  # Mix of x and y coordinates
p = len(coordinates) // 2  # Number of objects with such coordinates in 2D space
nFrames = 8
x_coordinates = np.zeros((p, nFrames), dtype=float)
y_coordinates = np.zeros((p, nFrames), dtype=float)
(rows, cols) = np.shape(x_coordinates)
# Special feeling of the initialized before matrix
for i in range(rows):
    for j in range(cols):
        x_coordinates[i, j] = float(coordinates[i*2])
# Special feeling of the initialized before matrix
for i in range(rows):
    for j in range(cols):
        y_coordinates[i, j] = float(coordinates[i*2+1])
