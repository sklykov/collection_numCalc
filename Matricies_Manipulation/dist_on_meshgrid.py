# -*- coding: utf-8 -*-
"""
Testing the distance calculation over the meshgrid.

@author: sklykov
"""
import numpy as np


def distance_f(i_px, j_px, i_centre: int, j_centre: int):
    """
    Calculate the distances for pixels.

    Parameters
    ----------
    i_px : int or numpy.ndarray
        Pixel(-s) i of an image.
    j_px : int or numpy.ndarray
        Pixel(-s) i of an image.
    i_centre : int
        Center of an image.
    j_centre : int
        Center of an image.

    Returns
    -------
    float or numpy.ndarray
        Distances between provided pixels and the center of an image.

    """
    return np.round(np.sqrt(np.power(i_px - i_centre, 2) + np.power(j_px - j_centre, 2)), 6)


x = np.arange(start=0, stop=3, step=1)
y = np.arange(start=0, stop=3, step=1)
msh = np.meshgrid(x, y)
distances = distance_f(msh[0], msh[1], 1, 1)
