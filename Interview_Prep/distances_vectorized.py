# -*- coding: utf-8 -*-
"""
Broadcasting + Vectorization instead of 2 nested loops for distances calculation.

@author: sklykov

@license: The Unlicense

"""
import numpy as np
from scipy.optimize import linear_sum_assignment

points1 = np.asarray([[1, 2], [3, 3], [5, 4], [3, 1]])  # points with some coordinates - like first detection / frame
points2 = np.asarray([[2, 2], [5, 6], [3, 4]])  # points in the 2nd detection / frame
# Note that points1.shape = 3, 2 and point2.shape = 2, 2, cannot compute directly diff = points1 - points2
diff = points1[:, None, :] - points2[None, :, :]  # broadcasting dimensions (None -> 1) for making direct computation
# Compute distance using last axis as dimension number
distances = np.round(np.linalg.norm(diff, axis=2), 3)  # Note how it is computed point to point
# Global optimal matching of points by using Hungarian algorithm
indices_A, indices_B = linear_sum_assignment(distances)
# Make the pairs of matched points + along with their distances
pairs = list(zip(points1[indices_A].tolist(), points2[indices_B].tolist(), distances[indices_A, indices_B]))
