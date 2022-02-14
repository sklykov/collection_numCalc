# -*- coding: utf-8 -*-
"""
Calculation of aberrations by using non- and aberrated wavefronts recorded by a Shack-Hartmann sensor.

According to the doctoral thesis by Antonello, J (2014): https://doi.org/10.4233/uuid:f98b3b8f-bdb8-41bb-8766-d0a15dae0e27

@author: ssklykov
"""

# %% Imports and globals
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# %% Open and process images

