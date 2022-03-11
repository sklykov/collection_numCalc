# -*- coding: utf-8 -*-
"""
Calculation of aberrations by using non- and aberrated wavefronts recorded by a Shack-Hartmann sensor.

According to the doctoral thesis by Antonello, J. (2014): https://doi.org/10.4233/uuid:f98b3b8f-bdb8-41bb-8766-d0a15dae0e27

@author: ssklykov
"""

# %% Imports and globals
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
from skimage.util import img_as_ubyte
from skimage.feature import peak_local_max
import time
from matplotlib.patches import Rectangle, Circle
from scipy import ndimage
from zernike_pol_calc import radial_polynomial, radial_polynomial_derivative_dr, triangular_function
from zernike_pol_calc import triangular_derivative_dtheta
from numpy.linalg import lstsq
plt.close('all')


# %% Function definitions
def check_img_coordinate(max_coordinate, coordinate):
    """
    Check that specified coordinate lays in the image height or width.

    Parameters
    ----------
    max_coordinate : float or int
        Height or width.
    coordinate : float or int
        Coordinate under revision.

    Returns
    -------
    Corrected coordinate.

    """
    if coordinate > max_coordinate:
        return max_coordinate
    elif coordinate < 0.0:
        return 0.0
    else:
        return coordinate


def get_localCoM_matrix(image: np.ndarray, min_dist_peaks: int = 15, threshold_abs: float = 2,
                        region_size: int = 16, plot: bool = False) -> np.array:
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
    plot : bool, optional
        If it's true, it allows plotting of found local peaks, boxes and CoMs. The default is True.

    Returns
    -------
    coms : np.array
        Calculated center of masses coordinates.

    """
    detected_centers = peak_local_max(image, min_distance=min_dist_peaks, threshold_abs=threshold_abs)
    (rows, cols) = image.shape
    if plot:
        plt.figure(); plt.imshow(image)  # Plot found local peaks
        plt.plot(detected_centers[:, 1], detected_centers[:, 0], '.', color="red")
        plt.title("Found local peaks")
    half_size = region_size // 2  # Half of rectangle area for calculation of CoM
    size = np.size(detected_centers, 0)  # Number of found local peaks
    coms = np.zeros((size, 2), dtype='float')  # Center of masses coordinates initialization
    if plot:
        plt.figure(); plt.imshow(image)  # Plot found regions for CoM calculations
        plt.title("Local regions for center of mass calculations")
    for i in range(size):
        x_left_upper = check_img_coordinate(cols, detected_centers[i, 1] - half_size)
        y_left_upper = check_img_coordinate(rows, detected_centers[i, 0] - half_size)
        # left_upper_corner = [x_left_upper, y_left_upper]
        if plot:
            # Plot found regions for CoM calculations
            plt.gca().add_patch(Rectangle((x_left_upper, y_left_upper),
                                          2*half_size, 2*half_size, linewidth=1,
                                          edgecolor='yellow', facecolor='none'))
        # CoMs calculation
        subregion = image[y_left_upper:y_left_upper+2*half_size, x_left_upper:x_left_upper+2*half_size]
        (coms[i, 0], coms[i, 1]) = ndimage.center_of_mass(subregion)
        coms[i, 0] += y_left_upper; coms[i, 1] += x_left_upper
    # Plot found CoMs
    if plot:
        plt.figure(); plt.imshow(image)
        plt.plot(coms[:, 1], coms[:, 0], '.', color="green")
        plt.title("Found center of masses")
    return coms


def get_integr_limits_circular_lenses(image: np.ndarray, aperture_radius: float = 15.0,
                                      min_dist_peaks: int = 15, threshold_abs: float = 2,
                                      unknown_geometry: bool = True, debug: bool = False):
    """
    Calculate integration limits on theta for circular lenses (sub-apertures) and polar coordinates of their centers.

    Parameters
    ----------
    image : np.ndarray
        The image acquired by a Shack-Hartmann sensor.
    aperture_radius : float, optional
        In pixels on the image. The default is 15.0.
    min_dist_peaks : int, optional
        Minimum distance of peaks on the image in pixels. The default is 15.
    threshold_abs : float, optional
        Absolute threshold intensity value as the starting value for peak defining. The default is 2.
    unknown_geometry : bool, optional
        If the actual sizes of subapertures and their centers relative to the image are unknown. The default is True.
    debug : bool, optional
        For plotting some of the calculated values (sub-apertures, angels, etc.). The default is False.

    Returns
    -------
    integration_limits : ndarray of floats with size (N sub-apertures, 2)
        Integration limits on theta.
    theta0 : ndarray of floats (N sub-apertures, 1)
        Angles theta of sub-aperture centers from the global polar coordinates on the frame (image).
    rho0 : ndarray of floats (N sub-apertures, 1)
        Radiuses r of sub-aperture centers from the global polar coordinates on the frame (image).

    """
    # Plotting the estimated circular apertures with the centers the same as found on the image local peaks
    # This is only for the unknown geometry applied
    if unknown_geometry:
        if debug:
            plt.figure(); plt.imshow(image)  # Plot found regions for CoM calculations
        (rows, cols) = image.shape
        if debug:
            plt.plot(cols//2, rows//2, '+', color="red")  # Plotting the center of an input image
            plt.title("Estimated circular apertures of mircolenses array")
        # Detected centers below are searched for on the aberrated image, so the shifts in CoMs should be calculated relatively to it
        detected_centers = peak_local_max(image, min_distance=min_dist_peaks, threshold_abs=threshold_abs)
        rho0 = np.zeros(np.size(detected_centers, 0), dtype='float')  # polar coordinate of a lens center
        theta0 = np.zeros(np.size(detected_centers, 0), dtype='float')  # polar coordinate of a lens center
        theta_a = np.zeros(np.size(detected_centers, 0), dtype='float')  # integration limit on theta (1)
        theta_b = np.zeros(np.size(detected_centers, 0), dtype='float')  # integration limit on theta (2)
        integration_limits = [(0.0, 0.0) for i in range(np.size(detected_centers, 0))]
        for i in range(np.size(detected_centers, 0)):
            # Plot found regions for CoM calculations
            if debug:
                plt.gca().add_patch(Circle((detected_centers[i, 1], detected_centers[i, 0]), aperture_radius,
                                           edgecolor='green', facecolor='none'))
            # Calculation of the boundaries of the integration intervals for equations from the thesis
            # Maybe, mine interpretation is wrong or too direct
            x_relative = detected_centers[i, 1] - cols//2  # relative to the center of an image
            y_relative = detected_centers[i, 0] - rows//2   # relative to the center of an image
            rho0[i] = np.sqrt(np.power(x_relative, 2) + np.power(y_relative, 2))  # radial coordinate of lens pupil
            # Below - manual recalculation of angles
            # !!!: axes X, Y => cols, rows on an image, center of polar coordinates - center of an image
            # !!!: angles are calculated relative to swapped X and Y axes, calculation below in grads, after translated to radians
            if (x_relative > 0.0) and (y_relative > 0.0):
                theta0[i] = np.arctan(x_relative/y_relative)*(180/np.pi)
            elif (y_relative < 0.0) and (x_relative > 0.0):
                theta0[i] = 180.0 - (np.arctan(x_relative/abs(y_relative))*(180/np.pi))
            elif (y_relative < 0.0) and (x_relative < 0.0):
                theta0[i] = 180.0 + (np.arctan(x_relative/y_relative)*(180/np.pi))
            elif (y_relative > 0.0) and (x_relative < 0.0):
                theta0[i] = 360.0 - (np.arctan(abs(x_relative)/y_relative)*(180/np.pi))
            elif (y_relative == 0.0) and (x_relative > 0.0):
                theta0[i] = 90.0
            elif (y_relative == 0.0) and (x_relative < 0.0):
                theta0[i] = 270.0
            elif (x_relative == 0.0) and (y_relative > 0.0):
                theta0[i] = 0.0
            elif (x_relative == 0.0) and (y_relative < 0.0):
                theta0[i] = 180.0
            theta_a[i] = np.round(theta0[i] - np.arctan(aperture_radius/rho0[i])*(180/np.pi), 3)
            theta_b[i] = np.round(theta0[i] + np.arctan(aperture_radius/rho0[i])*(180/np.pi), 3)
            integration_limits[i] = (theta_a[i], theta_b[i])
        # Plotting the calculated coordinates in polar projection for checking that all their centers defined correctly
        if debug:
            plt.figure()
            plt.axes(projection='polar')
            plt.plot(np.radians(theta0), rho0, '.')
            plt.tight_layout()
        theta0 = np.radians(theta0)
        integration_limits = np.radians(integration_limits)
        return (integration_limits, theta0, rho0)


def get_integr_limits_circles_coms(image: np.ndarray, coms: np.ndarray, aperture_radius: float = 15.0,
                                   debug: bool = False):
    """
    Calculate integration limits on theta for circular lenses (sub-apertures) and polar coordinates of their centers.

    Parameters
    ----------
    image : np.ndarray
        The image acquired by a Shack-Hartmann sensor.
    coms : np.ndarray
        Coordinates of found center of masses on the image, for avoiding searching again of local peaks.
    aperture_radius : float, optional
        Of each sub-aperture of lens, in pixels on the image. The default is 15.0.
    debug : bool, optional
        For plotting some of the calculated values (sub-apertures, angels, etc.). The default is False.

    Returns
    -------
    integration_limits : ndarray of floats with size (N sub-apertures, 2)
        Integration limits on theta.
    theta0 : ndarray of floats (N sub-apertures, 1)
        Angles theta of sub-aperture centers from the global polar coordinates on the frame (image).
    rho0 : ndarray of floats (N sub-apertures, 1)
        Radiuses r of sub-aperture centers from the global polar coordinates on the frame (image).

    """
    # Plotting the estimated circular apertures with the centers the same as found center of masses on non- or aberrated images
    if debug:
        plt.figure(); plt.imshow(image)  # Plot found regions for CoM calculations
    (rows, cols) = image.shape
    if debug:
        plt.plot(cols//2, rows//2, '+', color="red")  # Plotting the center of an input image
        plt.title("Estimated circular apertures of mircolenses array")
    # Detected centers below are searched for on the aberrated image, so the shifts in CoMs should be calculated relatively to it
    detected_centers = coms
    rho0 = np.zeros(np.size(detected_centers, 0), dtype='float')  # polar coordinate r of a sub-aperture (lens)
    theta0 = np.zeros(np.size(detected_centers, 0), dtype='float')  # polar coordinate theta of a sub-aperture (lens)
    theta_a = np.zeros(np.size(detected_centers, 0), dtype='float')  # integration limit on theta (1)
    theta_b = np.zeros(np.size(detected_centers, 0), dtype='float')  # integration limit on theta (2)
    integration_limits = [(0.0, 0.0) for i in range(np.size(detected_centers, 0))]
    for i in range(np.size(detected_centers, 0)):
        # Plot found regions for CoM calculations
        if debug:
            plt.gca().add_patch(Circle((detected_centers[i, 1], detected_centers[i, 0]), aperture_radius,
                                       edgecolor='green', facecolor='none'))
        # Calculation of the boundaries of the integration intervals for equations from the thesis
        # Maybe, mine interpretation is wrong or too direct
        x_relative = detected_centers[i, 1] - cols//2  # relative to the center of an image
        y_relative = detected_centers[i, 0] - rows//2   # relative to the center of an image
        rho0[i] = np.sqrt(np.power(x_relative, 2) + np.power(y_relative, 2))  # radial coordinate of lens pupil
        # Below - manual recalculation of angles - I still more convince with these calculations
        # !!!: axes X, Y => cols, rows on an image, center of polar coordinates - center of an image
        # !!!: angles are calculated relative to swapped X and Y axes, calculation below in grads, after translated to radians
        if (x_relative > 0.0) and (y_relative > 0.0):
            theta0[i] = np.arctan(x_relative/y_relative)*(180/np.pi)
        elif (y_relative < 0.0) and (x_relative > 0.0):
            theta0[i] = 180.0 - (np.arctan(x_relative/abs(y_relative))*(180/np.pi))
        elif (y_relative < 0.0) and (x_relative < 0.0):
            theta0[i] = 180.0 + (np.arctan(x_relative/y_relative)*(180/np.pi))
        elif (y_relative > 0.0) and (x_relative < 0.0):
            theta0[i] = 360.0 - (np.arctan(abs(x_relative)/y_relative)*(180/np.pi))
        elif (y_relative == 0.0) and (x_relative > 0.0):
            theta0[i] = 90.0
        elif (y_relative == 0.0) and (x_relative < 0.0):
            theta0[i] = 270.0
        elif (x_relative == 0.0) and (y_relative > 0.0):
            theta0[i] = 0.0
        elif (x_relative == 0.0) and (y_relative < 0.0):
            theta0[i] = 180.0
        theta_a[i] = np.round(theta0[i] - np.arctan(aperture_radius/rho0[i])*(180/np.pi), 3)
        theta_b[i] = np.round(theta0[i] + np.arctan(aperture_radius/rho0[i])*(180/np.pi), 3)
        integration_limits[i] = (theta_a[i], theta_b[i])
    # Plotting the calculated coordinates in polar projection for checking that all their centers defined correctly
    if debug:
        plt.figure()
        plt.axes(projection='polar')
        plt.plot(np.radians(theta0), rho0, '.')
        plt.tight_layout()
    theta0 = np.radians(theta0)
    integration_limits = np.radians(integration_limits)
    return (integration_limits, theta0, rho0)


def get_integr_limits_centralized_subapertures(image: np.ndarray, coms: np.ndarray, coms_shifts: np.ndarray,
                                               aperture_radius: float = 15.0, debug: bool = False) -> tuple:
    """
    Calculate integration limits on theta for sub-apertures and polar coordinates of their centers relatively to the found central aperture.

    The found central aperture is calculated relatively to the center of the global image recorded by a sensor.

    Parameters
    ----------
    image : np.ndarray
        The image acquired by a Shack-Hartmann sensor.
    coms : np.ndarray
        Coordinates of found center of masses on the image around the defined local peaks (focal spots formed by sub-apertures).
    coms_shifts : np.ndarray
        Shifts of center of masses between non- and aberrated recorded iamges by Shack-Hartman sensor.
    aperture_radius : float, optional
        Of each sub-aperture of lens, in pixels on the image. The default is 15.0.
    debug : bool, optional
        For plotting some of the calculated values (sub-apertures, angels, etc.). The default is False.

    Returns
    -------
    tuple
        composed of (integration_limits, theta0, rho0, subapertures_wt_central, coms_shifts_wt_central).
        integration_limits : ndarray of floats with size (N sub-apertures, 2)
            Integration limits on theta.
        theta0 : ndarray of floats (N sub-apertures, 1)
            Angles theta of sub-aperture centers from the global polar coordinates on the frame (image).
        rho0 : ndarray of floats (N sub-apertures, 1)
            Radiuses r of sub-aperture centers from the global polar coordinates on the frame (image).
        subapertures_wt_central: np.ndarray
            Calculated of center of masses around the local peaks, but without the central sub-aperture.
        coms_shifts_wt_central: np.ndarray
            Shifts of center of masses, but without calculated for the central sub-aperture.
    """
    # Plotting the estimated circular apertures with the centers the same as found center of masses on non- or aberrated images
    if debug:
        plt.figure(); plt.imshow(image)  # Plot found regions for CoM calculations
    (rows, cols) = image.shape
    x_img_center = cols//2; y_img_center = rows//2
    # The center for integration on sub-apertures - should be some central sub-aperture in their array
    # Detected centers below are searched for on the aberrated image, so the shifts in CoMs should be calculated relatively to it
    detected_centers = coms
    rho0 = np.zeros(np.size(detected_centers, 0)-1, dtype='float')  # polar coordinate r of a sub-aperture (lens)
    theta0 = np.zeros(np.size(detected_centers, 0)-1, dtype='float')  # polar coordinate theta of a sub-aperture (lens)
    theta_a = np.zeros(np.size(detected_centers, 0)-1, dtype='float')  # integration limits (lower) on theta (1)
    theta_b = np.zeros(np.size(detected_centers, 0)-1, dtype='float')  # integration limits (higher) on theta (2)
    subapertures_wt_central = np.zeros((np.size(detected_centers, 0)-1, 2), dtype='float')
    coms_shifts_wt_central = np.zeros((np.size(detected_centers, 0)-1, 2), dtype='float')
    integration_limits = [(0.0, 0.0) for i in range(np.size(detected_centers, 0)-1)]
    # From mine experience, the central sub-aperture should be close to the center of the image
    x_min_dist = cols; y_min_dist = rows; i_center_subaperture = -1
    min_distance = np.sqrt(np.power(x_min_dist, 2) + np.power(y_min_dist, 2))
    # Looking for minimal distance between sub-aperture(detected center) and saving its index
    for i in range(np.size(detected_centers, 0)):
        distance = np.sqrt(np.power((detected_centers[i, 1] - x_img_center), 2) + np.power((detected_centers[i, 0] - y_img_center), 2))
        if (min_distance > distance):
            i_center_subaperture = i; min_distance = distance
    x_central_subaperture = detected_centers[i_center_subaperture, 1]; y_central_subaperture = detected_centers[i_center_subaperture, 0]
    # Plot the found central sub-aperture
    if debug:
        plt.plot(x_central_subaperture, y_central_subaperture, '+', color="red")
        plt.plot(x_img_center, y_img_center, '+', color="blue")
        plt.title("Estimated circular apertures of mircolenses array")
    # Calculation of integration limits - below
    j = 0
    for i in range(np.size(detected_centers, 0)):
        # Plot found regions for CoM calculations
        if debug:
            plt.gca().add_patch(Circle((detected_centers[i, 1], detected_centers[i, 0]), aperture_radius,
                                       edgecolor='green', facecolor='none'))
        if i == i_center_subaperture:  # Deleting the central subaperture from the calculations
            continue
        else:
            # Calculation of the boundaries of the integration intervals for equations from the thesis
            x_relative = detected_centers[i, 1] - x_central_subaperture  # relative to the central sub-aperture
            y_relative = detected_centers[i, 0] - y_central_subaperture   # relative to the central sub-aperture
            rho0[j] = np.sqrt(np.power(x_relative, 2) + np.power(y_relative, 2))  # radial coordinate of lens pupil
            # Below - manual recalculation of angles - I still more convince with these calculations
            # !!!: axes X, Y => cols, rows on an image, center of polar coordinates - center of an image
            # !!!: angles are calculated relative to swapped X and Y axes, calculation below in grads, after translated to radians
            if (x_relative > 0.0) and (y_relative > 0.0):
                theta0[j] = np.arctan(x_relative/y_relative)*(180/np.pi)
            elif (y_relative < 0.0) and (x_relative > 0.0):
                theta0[j] = 180.0 - (np.arctan(x_relative/abs(y_relative))*(180/np.pi))
            elif (y_relative < 0.0) and (x_relative < 0.0):
                theta0[j] = 180.0 + (np.arctan(x_relative/y_relative)*(180/np.pi))
            elif (y_relative > 0.0) and (x_relative < 0.0):
                theta0[j] = 360.0 - (np.arctan(abs(x_relative)/y_relative)*(180/np.pi))
            elif (y_relative == 0.0) and (x_relative > 0.0):
                theta0[j] = 90.0
            elif (y_relative == 0.0) and (x_relative < 0.0):
                theta0[j] = 270.0
            elif (x_relative == 0.0) and (y_relative > 0.0):
                theta0[j] = 0.0
            elif (x_relative == 0.0) and (y_relative < 0.0):
                theta0[j] = 180.0
            theta_a[j] = np.round(theta0[j] - np.arctan(aperture_radius/rho0[j])*(180/np.pi), 3)
            theta_b[j] = np.round(theta0[j] + np.arctan(aperture_radius/rho0[j])*(180/np.pi), 3)
            integration_limits[j] = (theta_a[j], theta_b[j])
            subapertures_wt_central[j, 0] = detected_centers[i, 0]; subapertures_wt_central[j, 1] = detected_centers[i, 1]
            coms_shifts_wt_central[j, 0] = coms_shifts[i, 0]; coms_shifts_wt_central[j, 1] = coms_shifts[i, 1]
            j += 1
    # Plotting the calculated coordinates in polar projection for checking that all their centers defined correctly
    if debug:
        plt.figure()
        plt.axes(projection='polar')
        plt.plot(np.radians(theta0), rho0, '.')
        plt.tight_layout()
    theta0 = np.radians(theta0)
    integration_limits = np.radians(integration_limits)
    return (integration_limits, theta0, rho0, subapertures_wt_central, coms_shifts_wt_central)


def rho_ab(rho0: float, theta: float, theta0: float, aperture_radius: float) -> tuple:
    """
    Calculate of integration limits on sub-aperture (crossed by global radius from the polar coordinates).

    For the drawing - check mentioned dissertation.

    Parameters
    ----------
    rho0 : float
        R coordinate of the sub-aperture center (O).
    theta : float
        Current angle from the global polar coordinate (relative to the frame's center).
    theta0 : float
        Theta coordinate of the sub-aperture center (O).
    aperture_radius : float
        Radius of a sub-aperture.

    Returns
    -------
    tuple
        As (rho_a, rho_b) - the integration limits for further integration of Zernike polynomials.

    """
    cosin = np.cos(theta-theta0)
    cosinsq = cosin*cosin
    rho0sq = rho0*rho0
    apertureRsq = aperture_radius*aperture_radius
    # ???: for some reason, for big polar coordinates relative to radius aperture, the equation below becomes
    # complex (negative under np.sqrt). As the workaround - assume that under sqrt operation the actual value is positive
    if (rho0sq*(cosinsq - 1) + apertureRsq) < 0:
        # print("angle diff:", np.round((theta-theta0)*(180/np.pi), 1),
        #       "np.sqrt from", np.round((rho0sq*(cosinsq - 1) + apertureRsq), 2))  # Debugging
        rho_a = rho0*cosin - np.sqrt(abs(rho0sq*(cosinsq - 1) + apertureRsq))
        rho_b = rho0*cosin + np.sqrt(abs(rho0sq*(cosinsq - 1) + apertureRsq))
    else:
        rho_a = rho0*cosin - np.sqrt(rho0sq*(cosinsq - 1) + apertureRsq)
        rho_b = rho0*cosin + np.sqrt(rho0sq*(cosinsq - 1) + apertureRsq)
    return (rho_a, rho_b)


def rho_integral_funcX(r: float, theta: float, m: int, n: int) -> float:
    """
    Return 2 functions under the integration on X axis specified in the thesis.

    Parameters
    ----------
    r : float
        Polar coordinate (rho).
    theta : float
        Polar coordinate (theta).
    m : int
        Azimutal order of the Zernike polynomial.
    n : int
        Radial order of the Zernike polynomial.

    Returns
    -------
    float
        Function value.

    """
    derivRmn = radial_polynomial_derivative_dr(m, n, r)
    Rmn = radial_polynomial(m, n, r)
    Angularmn = triangular_function(m, theta)
    derivAngularmn = triangular_derivative_dtheta(m, theta)
    return ((derivRmn*r*Angularmn*np.cos(theta)), (Rmn*derivAngularmn*np.sin(theta)))


def rho_integral_funcY(r: float, theta: float, m: int, n: int) -> float:
    """
    Return 2 functions under the integration on Y axis specified in the thesis.

    Parameters
    ----------
    r : float
        Polar coordinate (rho).
    theta : float
        Polar coordinate (theta).
    m : int
        Azimutal order of the Zernike polynomial.
    n : int
        Radial order of the Zernike polynomial.

    Returns
    -------
    float
        Function value.

    """
    derivRmn = radial_polynomial_derivative_dr(m, n, r)
    Rmn = radial_polynomial(m, n, r)
    Angularmn = triangular_function(m, theta)
    derivAngularmn = triangular_derivative_dtheta(m, theta)
    return ((derivRmn*r*Angularmn*np.sin(theta)), (Rmn*derivAngularmn*np.cos(theta)))


def calc_integrals_on_apertures(integration_limits: np.ndarray, theta0: np.ndarray, rho0: np.ndarray, m: int, n: int,
                                n_polynomial: int = 1, aperture_radius: float = 15.0, n_steps: int = 50):
    """
    Calculate integrals using trapezodial rule specified in the mentioned thesis on X and Y axes for the specified Zernike polynomial.

    Parameters
    ----------
    integration_limits : np.ndarray
        Calculated previously on theta polar coordinate.
    theta0 : np.ndarray
        Polar coordinates theta of sub-aperture centers.
    rho0 : np.ndarray
        Polar coordinates r of sub-aperture centers.
    m : int
        Azimutal order of the Zernike polynomial.
    n : int
        Radial order of the Zernike polynomial.
    n_polynomial: int
        Number of polynomial for which the integration calculated. The default is 1.
    aperture_radius : float, optional
        Radius of sub-aperture in pixels on the image. The default is 15.0.
    n_steps : int, optional
        Number of integration steps for both integrals. The default is 50.

    Returns
    -------
    integral_values : ndarray with sizes (Number of sub-apertures, 2)
        Resulting integration values for each sub-aperture and for both X and Y axes.

    """
    # Integration on theta and rho - for X, Y integrals of Zernike polynonial made on the sub-aperture
    integral_values = np.zeros((len(integration_limits), 2), dtype='float')  # Doubled values - for X,Y axes
    # TODO: Calibration taking into account the wavelength, focal length should be implemented later
    calibration = 1.0
    # print("# of used integration steps:", n_steps)
    # For each sub-aperture below integration on (r, theta) of the Zernike polynomials
    for i_subaperture in range(len(integration_limits)):
        print(f"Integration for #{n_polynomial} polynomial started on {i_subaperture} subaperture out of {len(integration_limits)}")
        (theta_a, theta_b) = integration_limits[i_subaperture]  # Calculated previously integration steps on theta (radians)
        delta_theta = (theta_b - theta_a)/n_steps  # Step for integration on theta
        theta = theta_a  # initial value of theta
        # Integration over theta (trapezoidal rule)
        integral_sumX = 0.0; integral_sumY = 0.0
        # integral_sum_Zernike = 0.0
        for j_theta in range(n_steps+1):
            # Getting limits for integration on rho
            (rho_a, rho_b) = rho_ab(rho0[i_subaperture], theta, theta0[i_subaperture], aperture_radius)
            # Integration on rho for X and Y axis (trapezoidal formula)
            # !!!: Because of higher order Zernike start to depend heavily on the selected radius rho, then some calibration
            # should be performed. There are 3 options:1) To get the integral Integral(Integral(Zmn*rho*drho*dtheta))
            # in the range of sub-apertures sizes. 2) As it is on Wiki page about Zernike polynomials, the calibration
            # on the square of Zernike polynomials: Integral(Integral((Zmn^2)*rho*drho*dtheta)). BOTH TESTED!
            # 3) Option - attempt to resemble the dependency on rho order - since the derivative guides to loosing 1 order,
            # then calibration should be only calculated up to Rmn order (not Rmn*rho or Rmn*Rmn*rho))
            rho = rho_a  # Lower integration boundary
            delta_rho = (rho_b - rho_a)/n_steps
            integral_sum_rhoX1 = 0.0; integral_sum_rhoX2 = 0.0; integral_sum_rhoY1 = 0.0; integral_sum_rhoY2 = 0.0
            integral_Zernike_rho = 0.0
            for j_rho in range(n_steps+1):
                if (j_rho == 0) or (j_rho == n_steps):
                    (X1, X2) = rho_integral_funcX(rho, theta, m, n)
                    integral_sum_rhoX1 += 0.5*X1; integral_sum_rhoX2 += 0.5*X2  # on X axis
                    (Y1, Y2) = rho_integral_funcY(rho, theta, m, n)
                    integral_sum_rhoY1 += 0.5*Y1; integral_sum_rhoY2 += 0.5*Y2  # on Y axis
                    # radPol = radial_polynomial(m, n, rho)
                    integral_Zernike_rho += 0.5*radial_polynomial(m, n, rho)  # calibration - integration of radial part
                else:
                    (X1, X2) = rho_integral_funcX(rho, theta, m, n)
                    integral_sum_rhoX1 += X1; integral_sum_rhoX2 += X2
                    (Y1, Y2) = rho_integral_funcY(rho, theta, m, n)
                    integral_sum_rhoY1 += Y1; integral_sum_rhoY2 += Y2
                    # radPol = radial_polynomial(m, n, rho)
                    integral_Zernike_rho += radial_polynomial(m, n, rho)  # calibration - integration of radial part
                rho += delta_rho

            integral_sum_rhoX = (integral_sum_rhoX1 - integral_sum_rhoX2)
            integral_sum_rhoY = (integral_sum_rhoY1 + integral_sum_rhoY2)
            integral_sum_rhoX /= integral_Zernike_rho; integral_sum_rhoY /= integral_Zernike_rho
            # integral_sum_rhoX *= delta_rho; integral_sum_rhoY *= delta_rho; integral_Zernike_rho *= delta_rho

            if (j_theta == 0) and (j_theta == n_steps):
                integral_sumX += 0.5*integral_sum_rhoX; integral_sumY += 0.5*integral_sum_rhoY
                # triangPol = triangular_function(m, theta)
                # integral_sum_Zernike += 0.5*integral_Zernike_rho  # caluibration - integration angular part
            else:
                integral_sumX += integral_sum_rhoX; integral_sumY += integral_sum_rhoY
                # triangPol = triangular_function(m, theta)
                # integral_sum_Zernike += integral_Zernike_rho  # caluibration - integration angular part
            theta += delta_theta
        integral_sumX *= delta_theta; integral_sumY *= delta_theta  # End of integration on theta
        # integral_sumX /= integral_sum_Zernike; integral_sumY /= integral_sum_Zernike  # Calibration on the integrals of Zmn
        # ???: the results of calculation using this definitions are disappointing!

        # The final integral values should be also calibrated to focal and wavelengths
        integral_values[i_subaperture, 1] = calibration*integral_sumX  # Partially calibration on the area of sub-aperture
        integral_values[i_subaperture, 0] = calibration*integral_sumY  # Partially calibration on the area of sub-aperture
        integral_values = np.round(integral_values, 8)
    return integral_values


def calc_integrals_on_apertures_unit_circle(integration_limits: np.ndarray, theta0: np.ndarray, rho0: np.ndarray, m: int, n: int,
                                            n_polynomial: int = 1, aperture_radius: float = 15.0, n_steps: int = 50):
    """
    Calculate integrals using trapezodial rule inside the sub-apertures, which lies inside the unit circle.

    Parameters
    ----------
    integration_limits : np.ndarray
        Calculated previously on theta polar coordinate.
    theta0 : np.ndarray
        Polar coordinates theta of sub-aperture centers.
    rho0 : np.ndarray
        Polar coordinates r of sub-aperture centers.
    m : int
        Azimutal order of the Zernike polynomial.
    n : int
        Radial order of the Zernike polynomial.
    n_polynomial: int
        Number of polynomial for which the integration calculated. The default is 1.
    aperture_radius : float, optional
        Radius of sub-aperture in pixels on the image. The default is 15.0.
    n_steps : int, optional
        Number of integration steps for both integrals. The default is 50.

    Returns
    -------
    integral_values : ndarray with sizes (Number of sub-apertures, 2)
        Resulting integration values for each sub-aperture, they are listed for Y and X axes (relatively to image coordinate system).

    """
    integral_values = np.zeros((len(integration_limits), 2), dtype='float')  # Doubled values - for X,Y axes
    calibration = 1.0  # Calibration taking into account the wavelength, focal length should be implemented later
    rho_unit_calibration = np.max(rho0) + aperture_radius  # For making integration on rho on unit circle
    # For each sub-aperture below integration on (r, theta) of the Zernike polynomials
    for i_subaperture in range(len(integration_limits)):
        print(f"Integration for #{n_polynomial} polynomial started on {i_subaperture} subaperture out of {len(integration_limits)}")
        # Integration limits and steps on theta
        (theta_a, theta_b) = integration_limits[i_subaperture]  # Calculated previously integration steps on theta (radians)
        delta_theta = (theta_b - theta_a)/n_steps  # Step for integration on theta
        theta = theta_a  # initial value of theta
        # Getting limits for integration on rho
        (rho_a, rho_b) = rho_ab(rho0[i_subaperture], theta, theta0[i_subaperture], aperture_radius)
        # !!!: Another approach to integration on rho: the entire picture with sub-apertures will be accounted as unit-radius circle
        # For that, all rho values should be normilized to the maximum rho0 coordinate + radius_subaperture
        rho_a /= rho_unit_calibration; rho_b /= rho_unit_calibration
        # Integration over theta (trapezoidal rule)
        integral_sumX = 0.0; integral_sumY = 0.0
        for j_theta in range(n_steps+1):
            # Integration on rho for X and Y axis (trapezoidal formula)
            rho = rho_a  # Lower integration boundary
            delta_rho = (rho_b - rho_a)/n_steps
            integral_sum_rhoX1 = 0.0; integral_sum_rhoX2 = 0.0; integral_sum_rhoY1 = 0.0; integral_sum_rhoY2 = 0.0
            for j_rho in range(n_steps+1):
                if (j_rho == 0) or (j_rho == n_steps):
                    (X1, X2) = rho_integral_funcX(rho, theta, m, n)
                    integral_sum_rhoX1 += 0.5*X1; integral_sum_rhoX2 += 0.5*X2  # on X axis
                    (Y1, Y2) = rho_integral_funcY(rho, theta, m, n)
                    integral_sum_rhoY1 += 0.5*Y1; integral_sum_rhoY2 += 0.5*Y2  # on Y axis
                else:
                    (X1, X2) = rho_integral_funcX(rho, theta, m, n)
                    integral_sum_rhoX1 += X1; integral_sum_rhoX2 += X2
                    (Y1, Y2) = rho_integral_funcY(rho, theta, m, n)
                    integral_sum_rhoY1 += Y1; integral_sum_rhoY2 += Y2
                rho += delta_rho
            integral_sum_rhoX = (integral_sum_rhoX1 - integral_sum_rhoX2)  # From thesis equations
            integral_sum_rhoY = (integral_sum_rhoY1 + integral_sum_rhoY2)
            # End of integration on rho (i.e. r from polar coordinates)
            integral_sum_rhoX *= delta_rho; integral_sum_rhoY *= delta_rho
            if (j_theta == 0) and (j_theta == n_steps):
                integral_sumX += 0.5*integral_sum_rhoX; integral_sumY += 0.5*integral_sum_rhoY
            else:
                integral_sumX += integral_sum_rhoX; integral_sumY += integral_sum_rhoY
            theta += delta_theta
        integral_sumX *= delta_theta; integral_sumY *= delta_theta  # End of integration on theta
        # integral values should be also calibrated to the areas of sub-apertures (the thesis)
        # integral_sumX /= (np.pi*np.power((aperture_radius/rho_unit_calibration), 2))
        # integral_sumY /= (np.pi*np.power((aperture_radius/rho_unit_calibration), 2))
        # actually, the integral values should be calibrated to the each sub-aperture area - depedning on the integration limits
        # !!! But the values of them acually small - differences between rho_b and rho_a
        integral_sumX /= 0.5*(theta_b - theta_a)*(((rho_b*rho_b)-(rho_a*rho_a)))
        integral_sumY /= 0.5*(theta_b - theta_a)*(((rho_b*rho_b)-(rho_a*rho_a)))
        # The final integral values should be also calibrated to focal and wavelengths
        # Below - the question about X and Y axes - their order
        integral_values[i_subaperture, 0] = calibration*integral_sumX  # Not yet implemented calibration, not necessary now
        integral_values[i_subaperture, 1] = calibration*integral_sumY
        integral_values = np.round(integral_values, 8)
    return integral_values


def calc_integral_matrix_zernike(zernike_polynomials_list: list, integration_limits: np.ndarray, theta0: np.ndarray,
                                 rho0: np.ndarray, aperture_radius: float = 15.0, n_steps: int = 50, on_unit_circle: bool = True):
    """
    Wrap for calculation of integral values on sub-apertures performing on several Zernike polynomials.

    For shortening time of calculation, use the increasing order of listing of polynomials coefficients (like [(-1, 1), (1, 1)])

    Parameters
    ----------
    integration_limits : np.ndarray
        Calculated previously on theta polar coordinate.
    theta0 : np.ndarray
        Polar coordinates theta of sub-aperture centers.
    rho0 : np.ndarray
        Polar coordinates r of sub-aperture centers.
    m : int
        Azimutal order of the Zernike polynomial.
    n : int
        Radial order of the Zernike polynomial..
    aperture_radius : float, optional
        Radius of sub-aperture in pixels on the image. The default is 15.0.
    n_steps : int, optional
        Number of integration steps for both integrals. The default is 50
    on_unit_circle: bool, optional
        If True, then integration goes on the unit circle, in attempt to reproduce the thesis results. The default is False.

    Returns
    -------
    integral_matrix : ndarray with sizes (Number of sub-apertures, Number of Zernikes)
        Resulting integration values for each sub-aperture and for both X and Y axes and specified Zernike values.

    """
    n_rows = np.size(rho0, 0)
    n_cols = 2*len(zernike_polynomials_list)  # Because calculation needed for both X and Y axes
    integral_matrix = np.zeros((n_rows, n_cols), dtype='float')
    # Shortening the time of calculation because of symmetrical integrals over sub-apertures for (-1, 1) and (1, 1),
    # (-2, 2) and (2, 2), (-3, 3) and (-4, 4) - applying below reassignment
    symmetrical_substition = False; i_simmetry = -1
    for i in range(len(zernike_polynomials_list)):
        (m, n) = zernike_polynomials_list[i]
        if ((m == -1) and (n == 1)) or ((m == -2) and (n == 2)) or ((m == -3) and (n == 3)) or ((m == -4) and (n == 4)):
            i_simmetry = i; symmetrical_substition = True
        if symmetrical_substition:
            if ((m == 1) and (n == 1)) or ((m == 2) and (n == 2)) or ((m == 3) and (n == 3)) or ((m == 4) and (n == 4)):
                # Shortening the time of calculation because of symmetrical integrals over sub-apertures for (-1, 1) and (1, 1),
                # (-2, 2) and (2, 2), (-3, 3) and (-4, 4) - applying below reassignment
                integral_matrix[:, 2*i] = integral_matrix[:, 2*i_simmetry+1]
                integral_matrix[:, 2*i+1] = -integral_matrix[:, 2*i_simmetry]
                print("Shortening of calculation used")
            else:
                # Normal integrals calculation
                if on_unit_circle:
                    integral_values = calc_integrals_on_apertures_unit_circle(integration_limits, theta0, rho0, m, n, n_polynomial=i+1,
                                                                              aperture_radius=aperture_radius, n_steps=n_steps)
                else:
                    integral_values = calc_integrals_on_apertures(integration_limits, theta0, rho0, m, n, n_polynomial=i+1,
                                                                  aperture_radius=aperture_radius, n_steps=n_steps)
                integral_matrix[:, 2*i] = integral_values[:, 0]
                integral_matrix[:, 2*i+1] = integral_values[:, 1]
        else:
            # Normal integrals calculation
            if on_unit_circle:
                integral_values = calc_integrals_on_apertures_unit_circle(integration_limits, theta0, rho0, m, n, n_polynomial=i+1,
                                                                          aperture_radius=aperture_radius, n_steps=n_steps)
            else:
                integral_values = calc_integrals_on_apertures(integration_limits, theta0, rho0, m, n, n_polynomial=i+1,
                                                              aperture_radius=aperture_radius, n_steps=n_steps)
            integral_matrix[:, 2*i] = integral_values[:, 0]
            integral_matrix[:, 2*i+1] = integral_values[:, 1]
        print(f"Calculated {i+1} polynomial out of {len(zernike_polynomials_list)}")
    return integral_matrix


def get_polynomials_coefficients(integral_matrix: np.ndarray, coms_shifts: np.ndarray) -> np.ndarray:
    """
    Get the solution to the equation S = E*Alpha.

    Thhere S - CoMs shifts, E - integral matrix, Alpha - coefficients for decompisition of the wavefront on sum of Zernike polynomials.

    Parameters
    ----------
    integral_matrix : np.ndarray
        Integrals of Zernike polynomials on sub-apertures.
    coms_shifts : np.ndarray
        Shifts of center of masses coordinates for each focal spot of each sub-aperture.

    Returns
    -------
    alpha_coefficients : np.ndarray
        Coefficients for decomposition.

    """
    # alpha_coefficientsXY  = lstsq(integral_matrix, coms_shifts, rcond=1E-6)[0]  # Provides with solutions more than 1E-6, that is "0"
    # The matrices should be changed in sizes for calculation the alpha coefficients for each Zernike polynomial.
    # This made according to suggestion in the paper Dai G.-M., 1994
    n_subapertures = np.size(coms_shifts, 0); n_polynomials = np.size(integral_matrix, 1) // 2
    integral_matrix_swapped = np.zeros((2*n_subapertures, n_polynomials), dtype='float')
    coms_shifts_swapped = np.zeros(2*n_subapertures, dtype='float')
    for i in range(n_subapertures):
        coms_shifts_swapped[2*i] = coms_shifts[i, 0]  # X values of all polynomials
        coms_shifts_swapped[2*i+1] = coms_shifts[i, 1]  # Y values of all polynomials
        for j in range(n_polynomials):
            integral_matrix_swapped[2*i, j] = integral_matrix[i, 2*j]  # X values of all polynomials
            integral_matrix_swapped[2*i+1, j] = integral_matrix[i, 2*j+1]   # Y values of all polynomials
    alpha_coefficients = lstsq(integral_matrix_swapped, coms_shifts_swapped, rcond=1E-6)[0]  # Provides with solutions more than 1E-6

    return alpha_coefficients


def get_overall_coms_shifts(pics_folder: str = "pics", background_pic_name: str = "picBackground.png",
                            nonaberrated_pic_name: str = "nonAberrationPic.png", aberrated_pic_name: str = "aberrationPic.png",
                            min_dist_peaks: int = 15, threshold_abs: float = 2, region_size: int = 16,
                            substract_background: bool = True, plot_found_focal_spots: bool = False) -> tuple:
    """
    Calculate shifts of center of masses around the focal spots of each sub-apertures.

    It wraps several function above.

    Parameters
    ----------
    pics_folder : str, optional
        Path to the folder with pictures for processing. The default is "pics".
    background_pic_name : str, optional
        Picture name containing background. The default is "picBackground.png".
    nonaberrated_pic_name : str, optional
        Picture name containing non-aberrated (flat) wavefront. The default is "nonAberrationPic.png".
    aberrated_pic_name : str, optional
        Picture name containing aberrated wavefront. The default is "aberrationPic.png".
    min_dist_peaks : int, optional
        Minimal distance between two local peaks. The default is 15.
    threshold_abs : float, optional
        Absolute minimal intensity value of the local peak. The default is 2.
    region_size : int, optional
        Size of local rectangle, there the center of mass is calculated. The default is 16.
    substract_background : bool, optional
        If False, from the opened images the stored background picture won't be substracted. The default is True.
    plot_found_focal_spots : bool, optional
        If True, the images with found local peaks and sub-apertures will be plotted. The default is False.

    Raises
    ------
    Exception
        If the specified path to the folder with pictures doesn't exist or not a directory and if images there aren't actual files.

    Returns
    -------
    tuple
        (shifts of center of masses on (X, Y), selected center of masses for finding integration limits, selected image for depicting found
         integration limits).

    """
    # Open images on some specified folder
    if pics_folder == "pics":
        # Default folder in the repository
        absolute_path = os.path.join(os.getcwd(), pics_folder)
        backgroundPath = os.path.join(absolute_path, background_pic_name)
        nonaberratedPath = os.path.join(absolute_path, nonaberrated_pic_name)
        aberratedPath = os.path.join(absolute_path, aberrated_pic_name)
    else:
        if not(os.path.exists(pics_folder)):
            raise Exception("Specified Path doesn't exist")
        else:
            if not(os.path.isdir(pics_folder)):
                raise Exception("Specified object isn't a directory")
            else:
                # Looking for images on the specified path
                backgroundPath = os.path.join(pics_folder, background_pic_name)
                nonaberratedPath = os.path.join(pics_folder, nonaberrated_pic_name)
                aberratedPath = os.path.join(pics_folder, aberrated_pic_name)
                if (not(os.path.isfile(backgroundPath)) or not(os.path.isfile(nonaberratedPath)) or not(os.path.isfile(aberratedPath))):
                    raise Exception("Some of specified images are not a actual file, check path and their names")
    # Open the stored files and extracting the recorded background from the wavefronts
    background = (io.imread(backgroundPath, as_gray=True)); nonaberrated = (io.imread(nonaberratedPath, as_gray=True))
    aberrated = (io.imread(aberratedPath, as_gray=True))
    if substract_background:
        # Substracting from the recorded pictures (non- and aberrated) the recorded background
        diff_nonaberrated = (nonaberrated - background); diff_nonaberrated = img_as_ubyte(diff_nonaberrated)
        diff_aberrated = (aberrated - background); diff_aberrated = img_as_ubyte(diff_aberrated)
    else:
        # Substracting background as the minimal value on the picture and manually transferring the pictures into the U8 (ubyte) type
        # Additionally, stretch a bit the contrast
        diff_nonaberrated = nonaberrated - abs(np.min(nonaberrated)); diff_nonaberrated *= (255/np.max(diff_nonaberrated))
        diff_nonaberrated = np.uint8(diff_nonaberrated)
        if plot_found_focal_spots:
            plt.figure(); plt.imshow(diff_nonaberrated); plt.tight_layout(); plt.title("Non-aberrated")
        diff_aberrated = aberrated - abs(np.min(aberrated)); diff_aberrated *= (255/np.max(diff_aberrated))
        diff_aberrated = np.uint8(diff_aberrated)
        if plot_found_focal_spots:
            plt.figure(); plt.imshow(diff_aberrated); plt.tight_layout(); plt.title("Aberrated")
    # If the recorded background helps to enhance the contrast, then preserve the operation of background substraction
    if plot_found_focal_spots and substract_background:
        plt.figure(); plt.imshow(background); plt.tight_layout(); plt.title("Background")
        plt.figure(); plt.imshow(nonaberrated); plt.tight_layout(); plt.title("Non-aberrated")
        plt.figure(); plt.imshow(diff_nonaberrated); plt.tight_layout(); plt.title("Non-aberrated - Background")
        plt.figure(); plt.imshow(diff_aberrated); plt.tight_layout(); plt.title("Aberrated - Background")
    # Below all if statements - for substite default threshold values for normal starting value for the threshold of a local peak
    if not(substract_background):
        if (threshold_abs == 2.0):
            threshold_abs_nonaber = np.round(np.mean(diff_nonaberrated), 0) + 1
            threshold_abs_aber = np.round(np.mean(diff_aberrated), 0) + 1
        else:
            # Assign specified threshold values for searching of local peaks (including default values)
            threshold_abs_nonaber = threshold_abs; threshold_abs_aber = threshold_abs
    else:
        # If background substracted, then assign the specified threshold
        threshold_abs_nonaber = threshold_abs; threshold_abs_aber = threshold_abs
    # Plotting of results on an image for debugging (for coms_nonaberrated)
    # CoMs = center of masses, calculated in the areas around found local peaks
    coms_nonaberrated = get_localCoM_matrix(diff_nonaberrated, min_dist_peaks=min_dist_peaks, threshold_abs=threshold_abs_nonaber,
                                            region_size=region_size, plot=plot_found_focal_spots)
    coms_aberrated = get_localCoM_matrix(diff_aberrated, min_dist_peaks=min_dist_peaks, threshold_abs=threshold_abs_aber,
                                         region_size=region_size)
    # Found center of masses can be in different order and found CoMs can be different for non- and aberrated pictures
    if np.size(coms_nonaberrated, 0) == np.size(coms_aberrated, 0):
        # Before the sorting is used, that can cause the errors
        # Now - more precise algorithm for searching for neighbours between 2 matrices
        # Fixed: now the order of coms should be the same as the further calculation of integration limits
        coms_shifts = np.zeros((np.size(coms_aberrated, 0), 2), dtype='float')
        pic_integral_limits = diff_aberrated; coms_integral_limits = coms_aberrated
        # Below - the algorithm for searching for the right pair between CoMs on non - and aberrated image
        for i in range(np.size(coms_aberrated, 0)):
            j = 0
            # For finding real shift - only matching pair with minimum of shift between coordinates;
            # But recorded shift should be with the sign!!!
            diffY = abs(coms_aberrated[i, 0] - coms_nonaberrated[j, 0]); diffX = abs(coms_aberrated[i, 1] - coms_nonaberrated[j, 1])
            diffY_sign = (coms_aberrated[i, 0] - coms_nonaberrated[j, 0]); diffX_sign = (coms_aberrated[i, 1] - coms_nonaberrated[j, 1])
            if (diffX < (min_dist_peaks - 2) and diffY < (min_dist_peaks - 2)):
                coms_shifts[i, 0] = diffY_sign; coms_shifts[i, 1] = diffX_sign
            else:
                # Searching for the appropriate candidate (actual shift should lay in the ranges on both axes) from nonaberrated CoMs
                for j in range(1, np.size(coms_nonaberrated, 0)):
                    diffY = abs(coms_aberrated[i, 0] - coms_nonaberrated[j, 0])
                    diffY_sign = (coms_aberrated[i, 0] - coms_nonaberrated[j, 0])
                    diffX = abs(coms_aberrated[i, 1] - coms_nonaberrated[j, 1])
                    diffX_sign = (coms_aberrated[i, 1] - coms_nonaberrated[j, 1])
                    if (diffX < (min_dist_peaks - 2) and diffY < (min_dist_peaks - 2)):
                        coms_shifts[i, 0] = diffY_sign; coms_shifts[i, 1] = diffX_sign; break  # Stop if the candidate found
    else:
        # If the number of detected peaks is different for aberrated and non-aberrated pictures
        if np.size(coms_nonaberrated, 0) > np.size(coms_aberrated, 0):
            coms_shifts = np.zeros((np.size(coms_aberrated, 0), 2), dtype='float')
            pic_integral_limits = diff_aberrated; coms_integral_limits = coms_aberrated
            if plot_found_focal_spots:
                print("# of calculated local center of masses on non-aberrated > than on aberrated")
            for i in range(np.size(coms_aberrated, 0)):
                j = 0
                diffY = abs(coms_aberrated[i, 0] - coms_nonaberrated[j, 0]); diffX = abs(coms_aberrated[i, 1] - coms_nonaberrated[j, 1])
                diffY_sign = (coms_aberrated[i, 0] - coms_nonaberrated[j, 0])
                diffX_sign = (coms_aberrated[i, 1] - coms_nonaberrated[j, 1])
                if (diffX < (min_dist_peaks - 2) and diffY < (min_dist_peaks - 2)):
                    coms_shifts[i, 0] = diffY_sign; coms_shifts[i, 1] = diffX_sign
                else:
                    # Searching for the candidate (actual shift should lay in the ranges on both axes) from nonaberrated CoMs
                    for j in range(1, np.size(coms_nonaberrated, 0)):
                        diffY = abs(coms_aberrated[i, 0] - coms_nonaberrated[j, 0])
                        diffY_sign = (coms_aberrated[i, 0] - coms_nonaberrated[j, 0])
                        diffX = abs(coms_aberrated[i, 1] - coms_nonaberrated[j, 1])
                        diffX_sign = (coms_aberrated[i, 1] - coms_nonaberrated[j, 1])
                        if (diffX < (min_dist_peaks - 2) and diffY < (min_dist_peaks - 2)):
                            coms_shifts[i, 0] = diffY_sign; coms_shifts[i, 1] = diffX_sign; break  # Stop if the candidate found
        else:
            coms_shifts = np.zeros((np.size(coms_nonaberrated, 0), 2), dtype='float')
            pic_integral_limits = diff_nonaberrated; coms_integral_limits = coms_nonaberrated
            # !!!: Implement the logic if needed, no time now

    return (coms_shifts, coms_integral_limits, pic_integral_limits)


# %% Tests of functionality
# Calculation of shifts between non- and aberrated images, integrals of Zernike polynomials on sub-apertures
if __name__ == '__main__':
    # Get the shifts in center of masses and aberrated image with extracted background
    (coms_shifts, coms_integral_limits, pic_integral_limits) = get_overall_coms_shifts(plot_found_focal_spots=True)
    # (integration_limits, theta0, rho0) = get_integr_limits_circles_coms(pic_integral_limits, coms=coms_integral_limits, debug=False)
    (integration_limits, theta0, rho0,
     subapertures, coms_shifts) = get_integr_limits_centralized_subapertures(pic_integral_limits, coms_integral_limits, coms_shifts,
                                                                             aperture_radius=16.0, debug=True)
    # %% Testing of the decomposition of aberrations into the sum of Zernike polynomials
    t1 = time.time()
    zernikes_set = [(-3, 3), (3, 3)]
    # zernikes_set = [(-1, 1)]
    integral_matrix = calc_integral_matrix_zernike(zernikes_set, integration_limits, theta0, rho0,
                                                   aperture_radius=18.0, n_steps=40)
    t2 = time.time(); print(f"Integration of the Zernike polynomials ({zernikes_set}) takes:", np.round(t2-t1, 3), "s")
    alpha_coefficients = get_polynomials_coefficients(integral_matrix, coms_shifts)
