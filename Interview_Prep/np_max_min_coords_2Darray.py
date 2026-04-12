# -*- coding: utf-8 -*-
"""
Return indices of min / max values in 2D array.

@author: sklykov

@license: The Unlicense

"""
import numpy as np


img = np.asarray([[5, 5, 2], [1, 5, 2], [3, 5, 1]])
coords_max_val = np.argwhere(np.isclose(img, img.max()))  # safer variant than comparison
coords_min_val = np.argwhere(np.isclose(img, img.min()))
