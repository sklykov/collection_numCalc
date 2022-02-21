# -*- coding: utf-8 -*-
"""
Test calculated values (compare, checking precision, etc.).

@author: ssklykov
"""
# %% Imports
import os
import numpy as np

# %% Evalution of importance on number of integration steps for calculation of integrals on sub-apertures
calibration_folder = os.path.dirname(__file__)
Z50_file_name = "ZSHwfs50[(-3, 3), (-1, 3)].npy"; Z50_file_path = os.path.join(calibration_folder, Z50_file_name)
Z50 = np.load(Z50_file_path)
Z25_file_name = "ZSHwfs25[(-3, 3), (-1, 3)].npy"; Z25_file_path = os.path.join(calibration_folder, Z25_file_name)
Z25 = np.load(Z25_file_path)
diff_Z50_Z25 = np.absolute(Z50 - Z25)  # for accessing the influence on the precision calculations of decreasing number of integration steps
