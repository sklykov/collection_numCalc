# -*- coding: utf-8 -*-
"""
Compose tests for implemented modal phase wavefront reconstruction using decomposition to Zernike polynomials.

@author: ssklykov
"""
# %% Imports
from calc_zernikes_sh_wfs import get_overall_coms_shifts, calc_integral_matrix_zernike
from calc_zernikes_sh_wfs import get_polynomials_coefficients, get_integr_limits_centralized_subapertures
from zernike_pol_calc import plot_zps_polar, get_classical_polynomial_name
import os
import time
import numpy as np

# %% Type of calibration
shwfs = True; repo_pics = False

# %% Making calibration (integration of Zernike polynomials over sub-apertures) once and reading the integral matrix later
# zernikes_set = [(-1, 1)]
# zernikes_set = [(-1, 1), (1, 1)]
# zernikes_set = [(-2, 2), (0, 2), (2, 2)]
# zernikes_set = [(-3, 3), (-1, 3), (1, 3), (3, 3)]
# zernikes_set = [(-4, 4), (-2, 4), (0, 4), (2, 4), (4, 4)]
zernikes_set = [(-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2), (-3, 3), (-1, 3), (1, 3), (3, 3), (-4, 4), (-2, 4), (0, 4), (2, 4), (4, 4)]

# %% Tests on the recorded pictures from the Shack-Hartmann sensor
# Manual change the working directory to the folder with stored pictures, outside the repository
if shwfs:
    os.chdir(".."); os.chdir(".."); os.chdir("sh_wfs"); aberrated_pic_name = "AstigmatismPic0.png"
    plot = False; debug = False; aperture_radius = 10.0
    (coms_shifts, coms_integral_lim, pic_int_lim) = get_overall_coms_shifts(pics_folder=os.getcwd(),
                                                                            background_pic_name="backgroundPic2.png",
                                                                            nonaberrated_pic_name="nonAberrationPic2.png",
                                                                            aberrated_pic_name=aberrated_pic_name,
                                                                            min_dist_peaks=20, threshold_abs=55, region_size=18,
                                                                            substract_background=False, plot_found_focal_spots=plot)
    # (integration_limits, theta0, rho0) = get_integr_limits_circles_coms(pic_int_lim, coms=coms_integral_lim,
    #                                                                     aperture_radius=aperture_radius, debug=debug)
    (integration_limits, theta0, rho0,
     subapertures, coms_shifts) = get_integr_limits_centralized_subapertures(pic_int_lim, coms_integral_lim, coms_shifts,
                                                                             aperture_radius=aperture_radius, debug=debug)
    calibration_file_name = f"ZSHwfs50{zernikes_set}.npy"
    current_path = os.path.dirname(__file__)  # get path to the folder containing the script
    calibrations = os.path.join(current_path, "calibrations")
    calibration_path = os.path.join(calibrations, calibration_file_name)
    precalculated_zernikes = os.path.join(calibrations, "Calibration14ZernikesShH_AstigmatismPic0.npy")
    if not(os.path.exists(precalculated_zernikes)):
        if not(os.path.exists(calibration_path)):
            t1 = time.time()
            # n_steps defines speed of calculations, suboptimal number of steps = 60, see "calibrations_tests.py"
            integral_matrix = calc_integral_matrix_zernike(zernikes_set, integration_limits, theta0, rho0,
                                                           aperture_radius=aperture_radius, n_steps=30)
            np.save(calibration_path, integral_matrix); t2 = time.time()
            if np.round(t2-t1, 3) > 60:
                print(f"Integration of the Zernike polynomials ({zernikes_set}) takes:", np.round((t2-t1)/60, 1), "minutes")
            else:
                print(f"Integration of the Zernike polynomials ({zernikes_set}) takes:", np.round(t2-t1, 3), "s")
        else:
            integral_matrix = np.load(calibration_path)
    else:
        integral_matrix = np.load(precalculated_zernikes)
        alpha_coefficients = list(get_polynomials_coefficients(integral_matrix, coms_shifts))
        set14zernikes = [(-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2), (-3, 3), (-1, 3),
                         (1, 3), (3, 3), (-4, 4), (-2, 4), (0, 4), (2, 4), (4, 4)]
        for i in range(len(set14zernikes)):
            print(get_classical_polynomial_name(set14zernikes[i]), ":", np.round(alpha_coefficients[i], 4))
        plot_zps_polar(set14zernikes, title="reconstraction of " + "actual defocus",
                       tuned=True, alpha_coefficients=alpha_coefficients)

# %% Calibration of pictures shared in the repository
if repo_pics:
    plot = False; debug = True; aperture_radius = 14.0
    (coms_shifts, coms_integral_lim, pic_int_lim) = get_overall_coms_shifts(plot_found_focal_spots=plot)
    # (integration_limits, theta0, rho0) = get_integr_limits_circles_coms(pic_int_lim, coms_integral_lim,
    #                                                                     aperture_radius=aperture_radius, debug=debug)
    # Another calibration - using found central sub-aperture
    (integration_limits, theta0, rho0,
     subapertures, coms_shifts) = get_integr_limits_centralized_subapertures(pic_int_lim, coms_integral_lim, coms_shifts,
                                                                             aperture_radius=aperture_radius, debug=debug)
    calibration_file_name = f"Z{zernikes_set}.npy"
    current_path = os.path.dirname(__file__)  # get path to the folder containing the script
    calibrations = os.path.join(current_path, "calibrations")
    calibration_path = os.path.join(calibrations, calibration_file_name)
    precalculated_zernikes = os.path.join(calibrations, "Calibration14ZernikesRepoPics_unitcircle.npy")
    if not(os.path.exists(precalculated_zernikes)):
        if not(os.path.exists(calibration_path)):
            t1 = time.time()
            # n_steps defines speed of calculations, suboptimal number of steps = 60, see "calibrations_tests.py"
            integral_matrix = calc_integral_matrix_zernike(zernikes_set, integration_limits, theta0, rho0,
                                                           aperture_radius=aperture_radius, n_steps=60, on_unit_circle=True)
            np.save(calibration_path, integral_matrix); t2 = time.time()
            if np.round(t2-t1, 3) > 60:
                print(f"Integration of the Zernike polynomials ({zernikes_set}) takes:", np.round((t2-t1)/60, 1), "minutes")
            else:
                print(f"Integration of the Zernike polynomials ({zernikes_set}) takes:", np.round(t2-t1, 3), "s")
        else:
            integral_matrix = np.load(calibration_path)
    else:
        integral_matrix = np.load(precalculated_zernikes)
        alpha_coefficients = list(get_polynomials_coefficients(integral_matrix, coms_shifts))
        set14zernikes = [(-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2), (-3, 3), (-1, 3),
                         (1, 3), (3, 3), (-4, 4), (-2, 4), (0, 4), (2, 4), (4, 4)]
        for i in range(len(set14zernikes)):
            print(get_classical_polynomial_name(set14zernikes[i]), ":", np.round(alpha_coefficients[i], 4))
        plot_zps_polar(set14zernikes, title="reconstraction of pics from the open repository",
                       tuned=True, alpha_coefficients=alpha_coefficients)
        # Plotting the 14 Zernikes, assuming that the graph from the calculation gives the coefficients for Zernike polynomials
        repo_coefficients = [-0.987, 0.367, 0.341, -1.434, 0.843, -0.306, 0.192, 0.375, -0.277, 0.343, -0.159, 0.171, -0.037, -0.016]
        plot_zps_polar(set14zernikes, title="profile reconstructed by Jacopo", tuned=True, alpha_coefficients=repo_coefficients)
