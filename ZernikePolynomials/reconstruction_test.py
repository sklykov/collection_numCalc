# -*- coding: utf-8 -*-
"""
Compose tests for implemented modal phase wavefront reconstruction using decomposition to Zernike polynomials.

@author: ssklykov
"""
# %% Imports
from calc_zernikes_sh_wfs import get_overall_coms_shifts, get_integr_limits_circular_lenses, calc_integral_matrix_zernike
# import matplotlib.pyplot as plt
import os
import time
import numpy as np

# %% Making calibration (integration of Zernike polynomials over sub-apertures) once and reading the integral matrix later
# zernikes_set = [(-1, 1), (1, 1)]
zernikes_set = [(-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2)]
calibration_file_name = f"zernikes{zernikes_set}.npy"
(coms_shifts, aberrated_pic) = get_overall_coms_shifts()
# plt.imshow(aberrated_pic)
current_path = os.getcwd(); calibrations = os.path.join(current_path, "calibrations")
calibration_path = os.path.join(calibrations, calibration_file_name)
if not(os.path.exists(calibration_path)):
    t1 = time.time()
    (integration_limits, theta0, rho0) = get_integr_limits_circular_lenses(aberrated_pic)
    integral_matrix = calc_integral_matrix_zernike(zernikes_set, integration_limits, theta0, rho0, n_steps = 100)
    np.save(calibration_path, integral_matrix)
    t2 = time.time(); print(f"Integration of the Zernike polynomials ({zernikes_set}) takes:", np.round(t2-t1, 3), "s")
else:
    integral_matrix = np.load(calibration_path)


# %% Tests
