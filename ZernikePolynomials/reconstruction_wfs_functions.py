# -*- coding: utf-8 -*-
"""
Specification of calculation functions for reconstruction using non- and aberrated wavefronts recorded by a Shack-Hartmann sensor.

These functions are called in 'wfs_reconstruction_ui.py' for using it in the GUI controlling program.
According to the doctoral thesis by Antonello, J. (2014): https://doi.org/10.4233/uuid:f98b3b8f-bdb8-41bb-8766-d0a15dae0e27

@author: ssklykov
"""

# %% Imports and globals
import numpy as np
from skimage.feature import peak_local_max
import time
from matplotlib.patches import Rectangle, Circle
from scipy import ndimage
from zernike_pol_calc import radial_polynomial, radial_polynomial_derivative_dr, triangular_function
from zernike_pol_calc import triangular_derivative_dtheta, normalization_factor
from zernike_pol_calc import tabular_radial_polynomial, tabular_radial_derivative_dr  # for speeding up calculations
from numpy.linalg import lstsq
from calc_zernikes_sh_wfs import check_img_coordinate


# %% Function definitions
def get_localCoM_matrix(image: np.ndarray, axes_fig, min_dist_peaks: int = 15, threshold_abs: float = 55.0,
                        region_size: int = 16) -> np.array:
    """
    Calculate local center of masses in the region around of found peaks (maximums).

    Parameters
    ----------
    image : np.ndarray
        Shack-Hartmann image with recorded wavefront.
    min_dist_peaks : int, optional
        Minimal distance between two local peaks. The default is 15.
    threshold_abs : float, optional
        Absolute minimal intensity value of the local peak. The default is 2.
    region_size : int, optional
        Size of local rectangle, there the center of mass is calculated. The default is 16.

    Returns
    -------
    coms : np.array
        Calculated center of masses coordinates.

    """
    detected_centers = peak_local_max(image, min_distance=min_dist_peaks, threshold_abs=threshold_abs)
    (rows, cols) = image.shape
    # axes_fig.plot(detected_centers[:, 1], detected_centers[:, 0], '.', color="red")  # plot local peaks
    half_size = region_size // 2  # Half of rectangle area for calculation of CoM
    size = np.size(detected_centers, 0)  # Number of found local peaks
    coms = np.zeros((size, 2), dtype='float')  # Center of masses coordinates initialization
    for i in range(size):
        x_left_upper = check_img_coordinate(cols, detected_centers[i, 1] - half_size)
        y_left_upper = check_img_coordinate(rows, detected_centers[i, 0] - half_size)
        # Plot found regions for CoM calculations
        # axes_fig.add_patch(Rectangle((x_left_upper, y_left_upper), 2*half_size, 2*half_size,
        #                              linewidth=1, edgecolor='yellow', facecolor='none'))
        # CoMs calculation
        subregion = image[y_left_upper:y_left_upper+2*half_size, x_left_upper:x_left_upper+2*half_size]
        (coms[i, 0], coms[i, 1]) = ndimage.center_of_mass(subregion)
        coms[i, 0] += y_left_upper; coms[i, 1] += x_left_upper
    # Plot found CoMs
    # axes_fig.plot(coms[:, 1], coms[:, 0], '.', color="green")
    return coms


def get_integral_limits_nonaberrated_centers(axes_fig, picture_as_array: np.ndarray, threshold_abs: float = 55.0,
                                             aperture_radius: float = 15.0) -> tuple:
    """
    Calculate the center of masses of localized focal spots and also the integration limits for further modal wavefront reconstruction.

    Focal spots are recorded within the non-aberrated image (flat wavefront) from Shack-Hartmann sensor.

    Parameters
    ----------


    Returns
    -------
    tuple
        (Centers of masses for non-aberrated image without the central sub-aperture, non-aberrated image,
         Theta (polar) coordinates of sub-apertures, Rho (polar) coordinates of sub-apertures, Integration limits for sub-apertures).

    """
    # Use the specified radius of sub-apertures for calculation of parameters for CoMs defining
    min_dist_peaks = int(np.round(2*aperture_radius, 0)); region_size = min_dist_peaks
    # CoMs = center of masses, calculated in the areas around found local peaks
    coms_nonaberrated = get_localCoM_matrix(picture_as_array, axes_fig, min_dist_peaks=min_dist_peaks,
                                            threshold_abs=threshold_abs, region_size=region_size)
    # Searching for the central sub-aperture that should be close to the center of the image
    (rows, cols) = picture_as_array.shape; x_img_center = cols//2; y_img_center = rows//2
    x_min_dist = cols; y_min_dist = rows; i_center_subaperture = -1
    min_distance = np.sqrt(np.power(x_min_dist, 2) + np.power(y_min_dist, 2))
    # Looking for minimal distance between sub-apertures and the center of the frame and saving its index
    for i in range(np.size(coms_nonaberrated, 0)):
        distance = np.sqrt(np.power((coms_nonaberrated[i, 1] - x_img_center), 2)
                           + np.power((coms_nonaberrated[i, 0] - y_img_center), 2))
        if min_distance > distance:
            i_center_subaperture = i; min_distance = distance
    x_central_subaperture = coms_nonaberrated[i_center_subaperture, 1]
    y_central_subaperture = coms_nonaberrated[i_center_subaperture, 0]
    # Plotting the found center of image and central sub-aperture
    axes_fig.plot(x_central_subaperture, y_central_subaperture, '+', color="red")
    # axes_fig.plot(x_img_center, y_img_center, '+', color="blue")
    # Detected centers below are searched for on the aberrated image, so the shifts in CoMs should be calculated relatively to it
    rho0 = np.zeros(np.size(coms_nonaberrated, 0)-1, dtype='float')  # polar coordinate r of a sub-aperture (lens)
    theta0 = np.zeros(np.size(coms_nonaberrated, 0)-1, dtype='float')  # polar coordinate theta of a sub-aperture (lens)
    theta_a = np.zeros(np.size(coms_nonaberrated, 0)-1, dtype='float')  # integration limits (lower) on theta (1)
    theta_b = np.zeros(np.size(coms_nonaberrated, 0)-1, dtype='float')  # integration limits (higher) on theta (2)
    subapertures_wt_central = np.zeros((np.size(coms_nonaberrated, 0)-1, 2), dtype='float')
    integration_limits = [(0.0, 0.0) for i in range(np.size(coms_nonaberrated, 0)-1)]
    # Calculation of integration limits - below
    j = 0
    for i in range(np.size(coms_nonaberrated, 0)):
        # Plot found regions for CoM calculations
        if not(i == i_center_subaperture):
            axes_fig.add_patch(Circle((coms_nonaberrated[i, 1], coms_nonaberrated[i, 0]), aperture_radius,
                                      edgecolor='green', facecolor='none'))
        if i == i_center_subaperture:  # Deleting the central subaperture from the calculations
            continue
        else:
            # Calculation of the boundaries of the integration intervals for equations from the thesis
            x_relative = coms_nonaberrated[i, 1] - x_central_subaperture  # relative to the central sub-aperture
            y_relative = -coms_nonaberrated[i, 0] + y_central_subaperture   # relative to the central sub-aperture
            rho0[j] = np.sqrt(np.power(x_relative, 2) + np.power(y_relative, 2))  # radial coordinate of lens pupil
            # Calculation of integration limits for the angles theta
            theta0[j] = np.arctan2(y_relative, x_relative)*(180/np.pi)  # Calculation arctan with right quadrant selection in grads!
            # ??? Maybe redundant but seems better to make conversion to all positive values angles:
            if theta0[j] < 0.0:
                theta0[j] += 360.0   # all negative angles become positive
            theta_a[j] = np.round(theta0[j] - np.arctan(aperture_radius/rho0[j])*(180/np.pi), 3)  # calculation in grads!
            theta_b[j] = np.round(theta0[j] + np.arctan(aperture_radius/rho0[j])*(180/np.pi), 3)  # calculation in grads!
            integration_limits[j] = (theta_a[j], theta_b[j])
            subapertures_wt_central[j, 0] = coms_nonaberrated[i, 0]; subapertures_wt_central[j, 1] = coms_nonaberrated[i, 1]
            j += 1
    # Plot the found and used CoMs for calculations
    axes_fig.plot(subapertures_wt_central[:, 1], subapertures_wt_central[:, 0], '.', color="red")
    # Before the calculation for angles made in grads, below the transfer to radians for passing it further to trigonometric functions
    theta0 = np.radians(theta0); integration_limits = np.radians(integration_limits)
    return subapertures_wt_central, theta0, rho0, integration_limits


# %% Tests
if __name__ == "__main__":
    pass
