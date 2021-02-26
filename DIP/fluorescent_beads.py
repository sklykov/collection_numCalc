# -*- coding: utf-8 -*-
"""
Experiments with simulation of images of fluorescent beads

@author: ssklykov
"""
# %% General imports
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte

# %% class definition
class image_beads():
    pass


# %% General parameters
width = 1000
height = 1000

# %% Testing features
background = np.zeros((width, height), dtype='ubyte')  # Generation of backround (check definition of an image sizes)
background = img_as_ubyte(background)
plt.figure(1)
# Below - representation according to the documentation:
# plt.cm.gray - for representing gray values, aspect - for filling image values in a window
# origin - for adjusting origin of pixels (0, 0), extent - regulation of axis values
# extent = (-0.5, numcols-0.5, -0.5, numrows-0.5)) - for origin = 'lower' - documents
plt.imshow(background, cmap=plt.cm.gray, aspect='auto', origin='lower', extent=(0, height, 0, width))
