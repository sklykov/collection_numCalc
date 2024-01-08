# -*- coding: utf-8 -*-
"""
Tests of PSF calculations.

@author: sklykov
"""
# %% Imports
import numpy as np
from math import pi
import warnings
import matplotlib.pyplot as plt
from scipy.special import j1
from LoaderFile import loadSampleImage  # loading sample from 'resources' folder
from scipy.ndimage import convolve  # for making convolution
from skimage.restoration import richardson_lucy


# %% Physical parameters
wavelength = 0.55  # in micrometers
k = 2.0*pi/wavelength  # angular frequency
NA = 0.95  # microobjective property, ultimately NA = d/2*f, there d - aperture diameter, f - distance to the object (focal length)
# Note that ideal Airy pattern will be (2*J1(x)/x)^2, there x = k*NA*r, there r - radius in the polar coordinates on the image
pixel_size = 0.125  # in micrometers, physical length in pixels (um / pixels)
pixel2um_coeff = k*NA*pixel_size  # coefficient used for relate pixels to physical units
pixel2um_coeff_plot = k*NA*(pixel_size/10.0)  # coefficient used for better plotting with the reduced pixel size for preventing pixelated

# %% Testing parameters
show_psfs = False
show_convolution_deconvolution = True


# %% Ideal (theoretical) PSF convolution / deconvolution
def ideal_psf_image(r):
    """
    Calculate ideal Airy intensity distribution (normalized to 1.0), as the function 4.0*(J1(r)/r).

    Parameters
    ----------
    r : float or numpy.ndarray
        Radial distance for calculation. Note that 0.0 - is the special point for calculation.

    Raises
    ------
    ValueError
        If provided ndarray is not single dimensional vector.

    Returns
    -------
    float or numpy.ndarray
        Values of Airy intensity distribution.

    """
    if isinstance(r, list):
        r = np.asarray(r)  # conversion from list to numpy array
    if isinstance(r, int):
        r = float(r)
    if isinstance(r, float):
        if abs(round(r, 12)) == 0.0:  # check that the argument provided with 0 value
            return 1.0  # the lim (J1(x)/x) = 1/2 with x -> 0
        else:
            return 4.0*(pow(j1(r)/r, 2))
    elif isinstance(r, np.ndarray):
        if len(r.shape) == 1:  # 1D array
            psf_values = np.zeros(shape=r.shape)
            r = np.round(r, 12)  # for exclude small remainings and make them equal to zero
            zero_indices, = np.where(r == 0.0)  # find the zero values
            if zero_indices.shape[0] > 0:
                if zero_indices.shape[0] > 1:
                    __warn_message = "Provided values r contains two or more zero values, so the calculation will be slow"
                    warnings.warn(__warn_message)
                    for i in range(r.shape[0]):
                        if i not in zero_indices:
                            psf_values[i] = 4.0*np.power(j1(r[i])/r[i], 2)
                        else:
                            psf_values[i] = 1.0
                else:
                    zero_i = zero_indices[0]
                    if zero_i == 0:
                        psf_values[1:] = 4.0*np.power(j1(r[1:])/r[1:], 2); psf_values[0] = 1.0
                    elif zero_i == r.shape[0] - 1:
                        psf_values[:zero_i] = 4.0*np.power(j1(r[:zero_i])/r[:zero_i], 2); psf_values[zero_i] = 1.0
                    else:
                        psf_values[0:zero_i] = 4.0*np.power(j1(r[0:zero_i])/r[0:zero_i], 2); psf_values[zero_i] = 1.0
                        psf_values[zero_i+1:] = 4.0*np.power(j1(r[zero_i+1:])/r[zero_i+1:], 2)
            else:
                psf_values = j1(r)/r
            return psf_values
        else:
            raise ValueError(f"Please provide the 1 dimensional vector as input r values, instead of shape: {r.shape}")


def show_ideal_psf(size: int, calibration_coefficient: float):
    """
    Plot the Airy intensity distribution on the image with WxH: (size, size) and using coefficient between pixel and physical distance.

    Note the color map is viridis.

    Parameters
    ----------
    size : int
        Size of picture for plotting.
    calibration_coefficient : float
        Relation between distance in pixels and um (see parameters at the start lines of the script).

    Returns
    -------
    None.

    """
    if size % 2 == 0:
        size += 1  # make the image with odd sizes
    img = np.zeros((size, size), dtype=float)
    i_center = size//2; j_center = size//2
    for i in range(size):
        for j in range(size):
            pixel_dist = np.sqrt(np.power((i - i_center), 2) + np.power((j - j_center), 2))
            img[i, j] = ideal_psf_image(pixel_dist*calibration_coefficient)
    if img[0, 0] > np.max(img)/100:
        __warn_message = "The provided size for plotting PSF isn't sufficient for proper representation"
        warnings.warn(__warn_message)
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=plt.cm.viridis, aspect='auto', origin='lower', extent=(0, size, 0, size))
    plt.tight_layout()


def get_ideal_psf_kernel(calibration_coefficient: float) -> np.ndarray:
    """
    Calculate centralized matrix with PSF coefficients using Airy distribution.

    Parameters
    ----------
    calibration_coefficient : float
        Relation between pixels and distance (physical).

    Returns
    -------
    kernel : numpy.ndarray
        Matrix with PSF values.

    """
    # Define the kernel size, including even small intensity pixels
    max_size = int(round(10.0*(1.0/calibration_coefficient), 0)) + 1
    for i in range(max_size):
        if ideal_psf_image(i*calibration_coefficient) < 0.001:  # including the second max ring as well, from the property of Airy distribution
            break
    # Make kernel with odd sizes for precisely centering the kernel
    size = 2*i - 1
    if i % 2 == 0:
        size = i + 1
    kernel = np.zeros(shape=(size, size))
    i_center = size//2; j_center = size//2
    # Calculate the PSF kernel for usage in convolution operation
    for i in range(size):
        for j in range(size):
            pixel_dist = np.sqrt(np.power((i - i_center), 2) + np.power((j - j_center), 2))
            kernel[i, j] = ideal_psf_image(pixel_dist*calibration_coefficient)
    return kernel


def convolute_img_psf(img: np.ndarray, psf_kernel: np.ndarray) -> np.ndarray:
    """
    Convolute the provided image with PSF kernel as 2D arrays and return the convolved image with the same type as the original one.

    Parameters
    ----------
    img : numpy.ndarray
        Sample image, not colour.
    psf_kernel : numpy.ndarray
        Calculated PSF kernel.

    Returns
    -------
    convolved_img : numpy.ndarray
        Result of convolution (used scipy.ndimage.convolve).

    """
    img_type = img.dtype
    convolved_img = convolve(np.float32(img), psf_kernel, mode='reflect')
    conv_coeff = np.sum(psf_kernel)
    if conv_coeff > 0.0:
        convolved_img /= conv_coeff  # correct the convolution result by dividing to the kernel sum
    convolved_img = convolved_img.astype(dtype=img_type)  # converting convolved image to the initial image
    return convolved_img


def plot_convolution(img: np.ndarray, psf_kernel: np.ndarray) -> int:
    """
    Plot on 2 figures the initial sample image and convolved one (calculated by using convolute_img_psf function).

    Parameters
    ----------
    img : numpy.ndarray
        Sample image
    psf_kernel : numpy.ndarray
        Calculated PSF kernel.

    Returns
    -------
    int
        Randomly selected id for named plots for reusing it for further plotting figure naming.

    """
    unique_id = np.random.randint(low=0, high=501)
    plt.figure(f"Sample image {unique_id}"); plt.imshow(img, cmap=plt.cm.gray); plt.axis('off'); plt.tight_layout()
    conv_img = convolute_img_psf(img, psf_kernel)
    plt.figure(f"Convolved image {unique_id}"); plt.imshow(conv_img, cmap=plt.cm.gray); plt.axis('off'); plt.tight_layout()
    return unique_id


def deconvolve_img(convolved_img: np.ndarray, psf_kernel: np.ndarray, n_iterations: int = 20, keep_border_artifacts: bool = False) -> np.ndarray:
    """
    Wrap the skimage.restoration.richardson_lucy deconvolution function along with useful functionality.

    Parameters
    ----------
    convolved_img : numpy.ndarray
        Sample image, not colour.
    psf_kernel : numpy.ndarray
        Calculated PSF kernel, possibly approximated.
    n_iterations : int, optional
        Number of iterations for running skimage.restoration.richardson_lucy. The default is 20.
    keep_border_artifacts : bool, optional
        Flag for removing / keeping . The default is False.

    Returns
    -------
    restored_img : numpy.ndarray
        Deconvolved image.

    """
    conv_img_type = convolved_img.dtype; conv_img_max = np.max(convolved_img)
    h, w = convolved_img.shape; kernel_size = int((psf_kernel.shape[0] + 1) / 2)
    # Conversion of an input image to a float image
    if 'float' not in str(conv_img_type):
        convolved_img = convolved_img.astype(dtype=np.float32)
    # Performing deconvolution and removing artifacts
    convolved_img /= conv_img_max  # scaling of an image
    restored_img = richardson_lucy(image=convolved_img, psf=psf_kernel, num_iter=n_iterations)
    # Restoring input image type
    restored_img *= conv_img_max
    if 'float' not in str(conv_img_type):
        restored_img = restored_img.astype(dtype=conv_img_type)
    if not keep_border_artifacts:
        if h - 2*kernel_size < 4 or w - 2*kernel_size < 4:
            __warn_message = "The provided image is too small relative to the PSF kernel size and will contain only border artificts"
            warnings.warn(__warn_message)
        else:
            restored_img = restored_img[kernel_size:h-kernel_size, kernel_size: w - kernel_size]  # remove border artifacts
    return restored_img


# %% Tests
if __name__ == "__main__":
    # various radii specifications for testing psf calculation
    # r = [0.0, 0.0, 0.2, 0.3]
    # r = np.linspace(0.0, 1.5, num=150)
    # r = np.linspace(-1.0, 1.0, num=21)
    # r = np.linspace(0, 10, num=11); r *= pixel2um_coeff
    # psf = ideal_psf_image(r)

    # Plotting the results of PSF, convolution / deconvolution
    plt.close('all')
    if show_psfs:
        show_ideal_psf(17, pixel2um_coeff/2); show_ideal_psf(80, pixel2um_coeff_plot)
    sample_img = loadSampleImage(); psf_kernel = get_ideal_psf_kernel(pixel2um_coeff/2)
    blurred_img = convolute_img_psf(sample_img, psf_kernel); restored_img = deconvolve_img(blurred_img, psf_kernel)
    if show_convolution_deconvolution:
        plot_id = plot_convolution(sample_img, psf_kernel)
        plt.figure(f"Restored image {plot_id}"); plt.imshow(restored_img, cmap=plt.cm.gray); plt.axis('off'); plt.tight_layout()
    plt.show()
