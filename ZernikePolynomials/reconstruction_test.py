# -*- coding: utf-8 -*-
"""
Compose tests for implemented modal phase wavefront reconstruction using decomposition to Zernike polynomials.

@author: ssklykov
"""
# %% Imports
from calc_zernikes_sh_wfs import get_overall_coms_shifts, get_integr_limits_circles_coms, calc_integral_matrix_zernike
from calc_zernikes_sh_wfs import get_polynomials_coefficients
from zernike_pol_calc import plot_zps_polar
import matplotlib.pyplot as plt
import os
import time
import numpy as np

# %% Making calibration (integration of Zernike polynomials over sub-apertures) once and reading the integral matrix later
# zernikes_set = [(-1, 1)]
zernikes_set = [(-3, 3), (-1, 3)]
# zernikes_set = [(-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2)]
# zernikes_set = [(-3, 3), (-1, 3), (1, 3), (3, 3)]
# zernikes_set = [(-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2), (-3, 3), (-1, 3), (1, 3), (3, 3), (-4, 4), (-2, 4), (0, 4), (2, 4), (4, 4)]

# %% Tests on the recorded pictures from the Shack-Hartmann sensor
# Manual change the working directory to the folder with stored pictures, outside the repository
os.chdir(".."); os.chdir(".."); os.chdir("sh_wfs")
plot = False; debug = False
(coms_shifts, coms_integral_lim, pic_int_lim) = get_overall_coms_shifts(pics_folder=os.getcwd(),
                                                                        background_pic_name="backgroundPic2.png",
                                                                        nonaberrated_pic_name="nonAberrationPic2.png",
                                                                        aberrated_pic_name="DefocusPic0.png",
                                                                        min_dist_peaks=20, threshold_abs=55, region_size=18,
                                                                        substract_background=False, plot_found_focal_spots=plot)
(integration_limits, theta0, rho0) = get_integr_limits_circles_coms(pic_int_lim, coms=coms_integral_lim,
                                                                    aperture_radius=12, debug=debug)
calibration_file_name = f"ZSHwfs25{zernikes_set}.npy"
current_path = os.path.dirname(__file__)  # get path to the folder containing the script
calibrations = os.path.join(current_path, "calibrations")
calibration_path = os.path.join(calibrations, calibration_file_name)
precalculated_zernikes = os.path.join(calibrations, "Zernike14PolsIntegralMatrix.npy")
if not(os.path.exists(precalculated_zernikes)):
    if not(os.path.exists(calibration_path)):
        t1 = time.time()
        # n_steps defines speed of calculations
        integral_matrix = calc_integral_matrix_zernike(zernikes_set, integration_limits, theta0, rho0, n_steps=20)
        np.save(calibration_path, integral_matrix)
        t2 = time.time(); print(f"Integration of the Zernike polynomials ({zernikes_set}) takes:", np.round(t2-t1, 3), "s")
    else:
        integral_matrix = np.load(calibration_path)
else:
    integral_matrix = np.load(precalculated_zernikes)
    alpha_coefficients = list(get_polynomials_coefficients(integral_matrix, coms_shifts))
    plot_zps_polar(zernikes_set, title=f"sum of first {len(zernikes_set)} Zernike polynomials",
                   tuned=True, alpha_coefficients=alpha_coefficients)
