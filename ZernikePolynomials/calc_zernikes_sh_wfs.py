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
import os
from skimage.util import img_as_ubyte
from skimage.feature import peak_local_max
import time
from matplotlib.patches import Rectangle, Circle
from scipy import ndimage
from zernike_pol_calc import radial_polynomial, radial_polynomial_derivative_dr, triangular_function
from zernike_pol_calc import triangular_derivative_dtheta
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
    coms = np.zeros((size, 2), dtype=float)  # Center of masses coordinates initialization
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
    if unknown_geometry and debug:
        plt.figure(); plt.imshow(image)  # Plot found regions for CoM calculations
        (rows, cols) = image.shape
        plt.plot(cols//2, rows//2, '+', color="red")  # Plotting the center of an input image
        plt.title("Estimated circular apertures of mircolenses array")
        detected_centers = peak_local_max(image, min_distance=min_dist_peaks, threshold_abs=threshold_abs)
        rho0 = np.zeros(np.size(detected_centers, 0), dtype='float')  # polar coordinate of a lens center
        theta0 = np.zeros(np.size(detected_centers, 0), dtype='float')  # polar coordinate of a lens center
        theta_a = np.zeros(np.size(detected_centers, 0), dtype='float')  # integration limit on theta (1)
        theta_b = np.zeros(np.size(detected_centers, 0), dtype='float')  # integration limit on theta (2)
        integration_limits = [(0.0, 0.0) for i in range(np.size(detected_centers, 0))]
        for i in range(np.size(detected_centers, 0)):
            # Plot found regions for CoM calculations
            plt.gca().add_patch(Circle((detected_centers[i, 1], detected_centers[i, 0]), aperture_radius,
                                       edgecolor='green', facecolor='none'))
            # Calculation of the boundaries of the integration intervals for equations from the thesis
            # Maybe, mine interpretation is wrong or too direct
            x_relative = detected_centers[i, 1] - cols//2  # relative to the center of an image
            y_relative = detected_centers[i, 0] - rows//2   # relative to the center of an image
            rho0[i] = np.sqrt(np.power(x_relative, 2) + np.power(y_relative, 2))  # radial coordinate of lens pupil
            # Below - manual recalculation of angles
            # !!!: axes X, Y => cols, rows on an image, center of polar coordinates - center of an image
            # !!!: angles are calculated relative to swapped X and Y axes, calculation below in grads, in the translated to radians
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
        # Plotting the calculated coordinates in polar projection for checking that all sub-aperture centers defined correctly
        plt.figure()
        plt.axes(projection='polar')
        plt.plot(np.radians(theta0), rho0, '.')
        plt.tight_layout()
        theta0 = np.radians(theta0)
        integration_limits = np.radians(integration_limits)
        return (integration_limits, theta0, rho0)


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
    cosinsq = np.power(np.cos(theta-theta0), 2)
    rho0sq = rho0*rho0
    apertureRsq = aperture_radius*aperture_radius
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
                                aperture_radius: float = 15.0, n_steps: int = 50):
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
        Radial order of the Zernike polynomial..
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
    calibration = 1
    # For each sub-aperture below integration on (r, theta) of the Zernike polynomials
    for i_subaperture in range(len(integration_limits)):
        (theta_a, theta_b) = integration_limits[i_subaperture]  # Calculated previously integration steps on theta (radians)
        delta_theta = (theta_b - theta_a)/n_steps  # Step for integration on theta
        theta = theta_a  # initial value of theta
        # Integration over theta (trapezoidal rule)
        integral_sumX = 0.0; integral_sumY = 0.0
        for j_theta in range(n_steps+1):
            # Getting limits for integration on rho
            (rho_a, rho_b) = rho_ab(rho0[i_subaperture], theta, theta0[i_subaperture], aperture_radius)

            # Integration on rho for X and Y axis (trapezoidal formula)
            # !!!: Because of higher order Zernike start to depend heavily on the selected radius rho,
            # so the calibration should be performed depending on radius rho => calculate integral R(m, n) in the same interval
            # The integral is taken for calibration only on radial Zernike, because the integration goes on that or derivatives,
            # which is reducing the order on r (dR/dr => loosing 1 order on r)
            rho = rho_a  # Lower integration boundary
            delta_rho = (rho_b - rho_a)/n_steps
            integral_sum_rhoX1 = 0.0; integral_sum_rhoX2 = 0.0; integral_sum_rhoY1 = 0.0; integral_sum_rhoY2 = 0.0
            integral_calibration_rho = 0.0
            for j_rho in range(n_steps+1):
                if (j_rho == 0) or (j_rho == n_steps):
                    (X1, X2) = rho_integral_funcX(rho, theta, m, n)
                    integral_sum_rhoX1 += 0.5*X1; integral_sum_rhoX2 += 0.5*X2  # on X axis
                    (Y1, Y2) = rho_integral_funcY(rho, theta, m, n)
                    integral_sum_rhoY1 += 0.5*Y1; integral_sum_rhoY2 += 0.5*Y2  # on Y axis
                    integral_calibration_rho += 0.5*radial_polynomial(m, n, rho)  # calibration
                else:
                    (X1, X2) = rho_integral_funcX(rho, theta, m, n)
                    integral_sum_rhoX1 += X1; integral_sum_rhoX2 += X2
                    (Y1, Y2) = rho_integral_funcY(rho, theta, m, n)
                    integral_sum_rhoY1 += Y1; integral_sum_rhoY2 += Y2
                    integral_calibration_rho += radial_polynomial(m, n, rho)
                rho += delta_rho

            integral_sum_rhoX = (integral_sum_rhoX1 - integral_sum_rhoX2)
            integral_sum_rhoY = (integral_sum_rhoY1 + integral_sum_rhoY2)
            # integral_sum_rhoX *= delta_rho; integral_sum_rhoY *= delta_rho; integral_calibration_rho *= delta_rho
            integral_sum_rhoX /= integral_calibration_rho; integral_sum_rhoY /= integral_calibration_rho  # End of integration on rho(r)

            if (j_theta == 0) and (j_theta == n_steps):
                integral_sumX += 0.5*integral_sum_rhoX; integral_sumY += 0.5*integral_sum_rhoY
            else:
                integral_sumX += integral_sum_rhoX; integral_sumY += integral_sum_rhoY
            theta += delta_theta
        integral_sumX *= delta_theta; integral_sumY *= delta_theta  # End of integration on theta

        # ???: How to make calibration on the angle and should it be done?

        # The final integral values should be also calibrated to focal and wavelengths
        integral_values[i_subaperture, 0] = calibration*integral_sumX  # Partially calibration on the area of sub-aperture
        integral_values[i_subaperture, 1] = calibration*integral_sumY  # Partially calibration on the area of sub-aperture
        integral_values = np.round(integral_values, 8)
    return integral_values


def calc_integral_matrix_zernike(zernike_polynomials_list: list, integration_limits: np.ndarray, theta0: np.ndarray,
                                 rho0: np.ndarray, aperture_radius: float = 15.0, n_steps: int = 50):
    """
    Wrap for calculation of integral values on sub-apertures performing on several Zernike polynomials.

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

    Returns
    -------
    integral_matrix : ndarray with sizes (Number of sub-apertures, Number of Zernikes)
        Resulting integration values for each sub-aperture and for both X and Y axes and specified Zernike values.

    """
    n_rows = np.size(rho0, 0)
    n_cols = 2*len(zernike_polynomials_list)  # Because calculation needed for both X and Y axes
    integral_matrix = np.zeros((n_rows, n_cols), dtype='float')
    for i in range(len(zernike_polynomials_list)):
        (m, n) = zernike_polynomials_list[i]
        integral_matrix[:, 2*i] = calc_integrals_on_apertures(integration_limits, theta0, rho0, m, n)[:, 0]
        integral_matrix[:, 2*i+1] = calc_integrals_on_apertures(integration_limits, theta0, rho0, m, n)[:, 1]
    return integral_matrix


# %% Open and process test images (from https://github.com/jacopoantonello/mshwfs)
pics_folder = "pics"; absolute_path = os.path.join(os.getcwd(), pics_folder)
background = "picBackground.png"; nonaberrated = "nonAberrationPic.png"; aberrated = "aberrationPic.png"
backgroundPath = os.path.join(absolute_path, background); nonaberratedPath = os.path.join(absolute_path, nonaberrated)
aberratedPath = os.path.join(absolute_path, aberrated)
# Open the stored files and extracting the recorded background from the wavefronts
background = (io.imread(backgroundPath, as_gray=True)); nonaberrated = (io.imread(nonaberratedPath, as_gray=True))
aberrated = (io.imread(aberratedPath, as_gray=True))
diff_nonaberrated = (nonaberrated - background); diff_nonaberrated = img_as_ubyte(diff_nonaberrated)
diff_aberrated = (aberrated - background); diff_aberrated = img_as_ubyte(diff_aberrated)
# plt.figure(); plt.imshow(diff_nonaberrated); plt.figure(); plt.imshow(diff_aberrated)

# %% Calculation of shifts between non- and aberrated images, integrals of Zernike polynomials on sub-apertures
get_localCoM_matrix(diff_nonaberrated, plot=True)  # Plotting of results on an image for debugging
coms_nonaberrated = get_localCoM_matrix(diff_nonaberrated)
coms_aberrated = get_localCoM_matrix(diff_aberrated)
coms_nonaberrated = np.sort(coms_nonaberrated, axis=0); coms_aberrated = np.sort(coms_aberrated, axis=0)
diff_coms = coms_nonaberrated - coms_aberrated

t1 = time.time(); m = 1; n = 1
(integration_limits, theta0, rho0) = get_integr_limits_circular_lenses(diff_nonaberrated, debug=True)
integral_values = calc_integrals_on_apertures(integration_limits, theta0, rho0, m, n)
t2 = time.time(); print(f"Integration of the single Zernike polynomial ({m},{n}) takes:", np.round(t2-t1, 3), "s")

# %% Testing of the decomposition of aberrations into the sum of Zernike polynomials
t1 = time.time()
# zernikes_set = [(-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2), (-3, 3), (-1, 3), (1, 3), (3, 3)]
# zernikes_set = [(-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2)]
# zernikes_set = [(-2, 2), (2, 2), (0, 2)]
zernikes_set = [(-1, 1), (1, 1), (0, 2)]
integral_matrix = calc_integral_matrix_zernike(zernikes_set, integration_limits, theta0, rho0)
t2 = time.time(); print(f"Integration of the Zernike polynomials ({zernikes_set}) takes:", np.round(t2-t1, 3), "s")
