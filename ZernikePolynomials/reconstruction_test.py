# -*- coding: utf-8 -*-
"""
Compose tests for implemented modal phase wavefront reconstruction using decomposition to Zernike polynomials.

@author: ssklykov
"""
# %% Imports
from calc_zernikes_sh_wfs import get_overall_coms_shifts, get_integr_limits_circular_lenses, calc_integral_matrix_zernike
from calc_zernikes_sh_wfs import get_polynomials_coefficients
from zernike_pol_calc import plot_zps_polar
import os
import time
import numpy as np

# %% Making calibration (integration of Zernike polynomials over sub-apertures) once and reading the integral matrix later
# zernikes_set = [(-1, 1), (1, 1)]
# zernikes_set = [(-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2)]
# zernikes_set = [(-3, 3), (-1, 3), (1, 3), (3, 3)]
zernikes_set = [(-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2), (-3, 3), (-1, 3), (1, 3), (3, 3), (-4, 4), (-2, 4), (0, 4), (2, 4), (4, 4)]

calibration_file_name = f"Z{zernikes_set}.npy"
(coms_shifts, aberrated_pic) = get_overall_coms_shifts()
# plt.imshow(aberrated_pic)
current_path = os.getcwd(); calibrations = os.path.join(current_path, "calibrations")
calibration_path = os.path.join(calibrations, calibration_file_name)
precalculated_zernikes = os.path.join(calibrations, "Zernike14PolsIntegralMatrix.npy")
if not(os.path.exists(precalculated_zernikes)):
    if not(os.path.exists(calibration_path)):
        t1 = time.time()
        (integration_limits, theta0, rho0) = get_integr_limits_circular_lenses(aberrated_pic)
        integral_matrix = calc_integral_matrix_zernike(zernikes_set, integration_limits, theta0, rho0, n_steps=200)
        np.save(calibration_path, integral_matrix)
        t2 = time.time(); print(f"Integration of the Zernike polynomials ({zernikes_set}) takes:", np.round(t2-t1, 3), "s")
    else:
        integral_matrix = np.load(calibration_path)
else:
    integral_matrix = np.load(precalculated_zernikes)
    alpha_coefficients = list(get_polynomials_coefficients(integral_matrix, coms_shifts))
    plot_zps_polar(zernikes_set, title=f"sum of first {len(zernikes_set)} Zernike polynomials",
                   tuned=True, alpha_coefficients=alpha_coefficients)


# %% Tests
