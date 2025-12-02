# coding=utf-8
"""
Fitting parameters of paraboloid over 2 dimensions.

@author: sklykov

@license: The Unlicense

"""
# %% Global imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from typing import Callable

matplotlib.use('Qt5Agg')  # universally works for PyCharm and Spyder for setting interactive plotting

# Dev. Note: np.linalg.lstsq does not accept or use any initial guess — because least-squares for a linear model has
# a closed-form analytic solution. There is no iteration, no starting point, no optimization. (Conceptual!)


# %% Func. def-s
def paraboloid(coefficients: np.ndarray, coordinates: tuple) -> float:
    """
    Return a paraboloid z value, based on equation a*x^2 + b*y^2 + c*x*y + d*x + e*y + f.

    Parameters
    ----------
    coefficients : np.ndarray
        [a, b, c, d, e, f].
    coordinates : tuple
        As tuple (x, y) and x, y could be also vectors.

    Returns
    -------
    float
        z[x, y] value.

    """
    x, y = coordinates[0], coordinates[1]
    a, b, c, d, e, f = coefficients[0], coefficients[1], coefficients[2],  coefficients[3], coefficients[4], coefficients[5]
    return a*x*x + b*y*y + c*x*y + d*x + e*y + f


def fit_paraboloid(x_flat: np.ndarray, y_flat: np.ndarray, paraboloid_f: Callable) -> np.ndarray:
    """
    Find paraboloid coefficients by solving linear least square matrix equation.

    Parameters
    ----------
    x_flat : np.ndarray
        Flattend x coordinates (can be obtained from a meshgrid).
    y_flat : np.ndarray
        Flattend y coordinates (can be obtained from a meshgrid).
    paraboloid_f : Callable
        Computing z surface of paraboloid surface.

    Returns
    -------
    p : np.ndarray
        6 paraboloid coefficients: a, b, c, d, e, f.

    """
    A = np.column_stack([x_flat*x_flat, y_flat*y_flat, x_flat*y_flat, x_flat, y_flat, np.ones_like(x_flat)])
    p = [0.0]*6  # a, b, c, d, e, f
    try:
        p, *_ = np.linalg.lstsq(A, z, rcond=None)
    except np.linalg.LinAlgError:
        __message = "Paraboloid not fitted for provided parameters! All zeros will be returned!"; warnings.warn(__message)
    return p


def define_paraboloid_peak(coefficients: np.ndarray) -> tuple:
    a, b, c, d, e, f = coefficients  # unpacking provided coefficients
    # Find peak according to computed gradient (dz/dx) and relation that it's zero in extremum: dz/dx = 0, dz/dy = 0
    h = np.asarray([[2*a, c], [c, 2*b]])  # Actually, Hessian matrix
    b = np.asarray([-d, -e])
    peak_coords, *_ = np.linalg.lstsq(h, b, rcond=None)
    # Check that peak is maximum using Hessian matrix
    is_maximum = h[0, 0] < 0.0 and h[1, 1] < 0.0 and h[0, 0]*h[1, 1] - h[0, 1]*h[1, 0] > 0.0
    return (peak_coords, is_maximum)


# %% Run as a script
if __name__ == "__main__":
    plt.close("all")
    # From Chat: a = -2; b = -3; c = 0.5; d = 0.75; e = -0.8; f = 0 => peak at (0.2, -0.1)
    coeffs = [-2.0, -3.0, 0.5, 0.75, -0.8, 1.0]  # paraboloid parameters
    x_ampl = y_ampl = 0.5; coord_step = 0.1
    x = np.round(np.arange(start=-x_ampl, stop=x_ampl + coord_step, step=coord_step), 2)
    y = np.round(np.arange(start=-y_ampl, stop=y_ampl + coord_step, step=coord_step), 2)
    x_coords, y_coords = np.meshgrid(x, y)  # makes 2D matrix for meshgrid
    x_coords_fl = x_coords.ravel(); y_coords_fl = y_coords.ravel()  # flatten
    z = np.empty_like(x_coords_fl); i = 0
    for x_c, y_c in zip(x_coords_fl, y_coords_fl):  # flattens input meshgrids and makes a tuple from them
        z[i] = round(paraboloid(coeffs, (x_c, y_c)), 4); i += 1
    z_surf = z.reshape(x_coords.shape)
    # Code above produces same result as:
    # z = np.empty_like(x_coords)
    # for i in range(x_coords.shape[0]):
    #     for j in range(x_coords.shape[1]):
    #         z[i, j] = paraboloid(coeffs, (x_coords[i, j], y_coords[i, j]))
    # z = np.round(z, 4)

    # Perform fit procedure using linear for defining coefficients of paraboloid
    coeffs_fitted = np.round(fit_paraboloid(x_coords_fl, y_coords_fl, z), 6)
    peak_coords, is_peak_max = define_paraboloid_peak(coeffs_fitted)
    peak_coords = np.round(peak_coords, 2)
    print("Defined peak coordinates (0.2, -0.1) and found by fitting:", peak_coords)

    # Plotting call
    fig = plt.figure(); axes = fig.add_subplot(projection='3d')
    axes.plot_surface(x_coords, y_coords, z_surf, cmap='viridis'); plt.tight_layout()
    plt.show()
