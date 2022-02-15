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
                                      min_dist_peaks: int = 12, threshold_abs: float = 2,
                                      unknown_geometry: bool = True, debug: bool = False):
    # Plotting the estimated circular apertures with the centers the same as found on the image local peaks
    # This is only for the unknown geometry applied
    if unknown_geometry and debug:
        plt.figure(); plt.imshow(image)  # Plot found regions for CoM calculations
        (rows, cols) = image.shape
        plt.plot(cols//2, rows//2, '+', color="red")  # Plotting the center of an input image
        plt.title("Estimated circular apertures of mircolenses array")
        detected_centers = peak_local_max(image, min_distance=min_dist_peaks, threshold_abs=threshold_abs)
        for i in range(np.size(detected_centers, 0)):
            # Plot found regions for CoM calculations
            plt.gca().add_patch(Circle((detected_centers[i, 1], detected_centers[i, 0]), aperture_radius,
                                       edgecolor='green', facecolor='none'))


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

# %% Shifts between aberrated and non-aberrated pictures calculation
get_localCoM_matrix(diff_nonaberrated, plot=True)  # Plotting of results on an image for debugging
t1 = time.time()
coms_nonaberrated = get_localCoM_matrix(diff_nonaberrated)
coms_aberrated = get_localCoM_matrix(diff_aberrated)
coms_nonaberrated = np.sort(coms_nonaberrated, axis=0); coms_aberrated = np.sort(coms_aberrated, axis=0)
diff_coms = coms_nonaberrated - coms_aberrated
t2 = time.time()
print("Shift of local center of masses between non- and aberrated images takes: ", np.round(t2-t1, 3), "s")
get_integr_limits_circular_lenses(diff_nonaberrated, debug=True)
