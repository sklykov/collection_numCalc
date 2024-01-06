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

# %% Physical parameters
wavelength = 0.55  # in micrometers
k = 2*pi/wavelength  # angular frequency
NA = 1.0  # microobjective property, ultimately NA = d/2*f, there d - aperture diameter, f - distance to the object (focal length)
# Note that ideal Airy pattern will be (2*J1(x)/x)^2, there x = k*NA*r, there r - radius in the polar coordinates on the image
pixel_size = 0.15  # in micrometers, physical length in pixels (um / pixels)
pixel2um_coeff = k*NA*pixel_size  # coefficient used for relate pixels to physical units


# %% Functions
def ideal_psf_image(r):
    if isinstance(r, list):
        r = np.asarray(r)  # conversion from list to numpy array
    if isinstance(r, float):
        if abs(round(r, 12)) == 0.0:  # check that the argument provided with 0 value
            return 1.0  # the lim (J1(x)/x) = 1/2 with x -> 0
        else:
            return 2.0*(pow(j1(r)/r, 2))
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
                        psf_values[zero_i+1:] = 4.0*(j1(r[zero_i+1:])/r[zero_i+1:])
            else:
                psf_values = j1(r)/r
            return psf_values
        else:
            raise ValueError(f"Please provide the 1 dimensional vector as input r values, instead of shape: {r.shape}")


# %% Tests
if __name__ == "__main__":
    # r = [0.0, 0.0, 0.2, 0.3]
    # r = np.linspace(0.0, 1.5, num=150)
    # r = np.linspace(-1.0, 1.0, num=21)
    r = np.linspace(0, 10, num=11); r *= pixel2um_coeff
    psf = ideal_psf_image(r)
