# -*- coding: utf-8 -*-
"""
Evaluation of curve fitting by least square method (from scipy.optimize module).

@author: sklykov

@license: The Unlicense

"""
# %% Imports
import numpy as np
from scipy.optimize import curve_fit
from typing import Union
import matplotlib.pyplot as plt
import random


# %% Functions defs.
def lorentzian(x: np.ndarray, x0: Union[float, int], gamma: float) -> np.ndarray:
    """
    Generate Lorentzian curve (Cauchy PDF).

    Parameters
    ----------
    x : np.ndarray
        Sample data.
    x0 : Union[float, int]
        Median value.
    gamma : float
        See the definition.

    Returns
    -------
    np.ndarray
        Lorentzian.

    """
    return gamma/(gamma*gamma + np.power(x-x0, 2))


def line_func(x: np.ndarray, k: Union[float, int], b: Union[float, int]) -> np.ndarray:
    """
    Line function k*x + b.

    Parameters
    ----------
    x : np.ndarray
        Sample data.
    k : Union[float, int]
        k coefficient.
    b : Union[float, int]
        additional value.

    Returns
    -------
    np.ndarray
        Line from equation k*x + b.

    """
    return k*x + b


def add_noise_each_point(y: np.ndarray, percentage: int = 10) -> np.ndarray:
    """
    Add uniformly deviated noise value (+/-) with the percentage of amplitude.

    Parameters
    ----------
    y : np.ndarray
        Vector of values.
    percentage : int, optional
        Percentage of amplitude that will be added / substracted from the original value. The default is 10.

    Returns
    -------
    y_noised : np.ndarray
        Original vector +- noise (uniformly distributed).

    """
    y_noised = np.copy(y); noise_percent = abs(percentage) / 100
    for i in range(y.shape[0]):
        sign_i = random.choice([-1.0, 1.0])
        if abs(y[i]) > 1.0:
            y_min = -y[i]*noise_percent; y_max = y[i]*(noise_percent + 0.0025)
        else:
            y_min = -noise_percent; y_max = noise_percent
        y_noised[i] = y[i] + sign_i*random.uniform(a=y_min, b=y_max)
    return y_noised


def parabola_fit_func(x: Union[float, np.ndarray], a: float, b: float, c: float) -> Union[float, np.ndarray]:
    """
    Callable parabola function for fitting.

    Parameters
    ----------
    x : Union[float, np.ndarray]
        Function values.
    a : float
        Coefficient on x^2.
    b : float
        Coefficient on x.
    c : float
        Coefficient on x^0.

    Returns
    -------
    Union[float, np.ndarray]
        Values of function y = a*x^2 + b*x + c.

    """
    return a*x*x + b*x + c


def gaussian_fit_func(x: Union[float, np.ndarray], a: float, b: float, c: float) -> Union[float, np.ndarray]:
    """
    Parametric Gaussian function for fitting.

    Equation: a*exp(-(x-b)^2/2*c^2)

    Parameters
    ----------
    x : Union[float, np.ndarray]
        Function values.
    a : float
        See equation.
    b : float
        See equation. Mean value.
    c : float
        See equation. Sigma value.

    Returns
    -------
    Union[float, np.ndarray]
        a*exp(-(x-b)^2/2*c^2).

    """
    return a*np.exp(-np.power(x-b, 2)/(2.0*np.power(c, 2)))


def lorentzian_fit_func(x: Union[float, np.ndarray], a: float, b: float) -> Union[float, np.ndarray]:
    """
    Parametric Lorentzian (Cauchy PDF) function for fitting.

    Equation: a/(a*a + (x-b)^2)

    Parameters
    ----------
    x : Union[float, np.ndarray]
        Function values.
    a : float
        See equation (gamma value).
    b : float
        See equation. Mean value (x0).

    Returns
    -------
    Union[float, np.ndarray]
        a*exp(-(x-b)^2/2*c^2).

    """
    return a/(a*a + np.power(x-b, 2))


# %% Sample data generation (lorentzian with noise)
x_data = np.linspace(start=-2.0, stop=2.0, num=41)  # arbitrary X data
y_lor = lorentzian(x_data, x0=0.0, gamma=1.0)  # lorentzian function
y_lor_noise = add_noise_each_point(y_lor, percentage=20)  # adding noise value to each data point sequentially from uniform distribution
# Parabola fit
try:
    fitted_parabola_params = curve_fit(parabola_fit_func, x_data, y_lor_noise)[0]
except RuntimeError:
    fitted_parabola_params = None; print("Fit to Parabola failed")
if fitted_parabola_params is not None:
    y_parabola_fit = parabola_fit_func(x_data, *fitted_parabola_params)
# Gaussian fit
try:
    fitted_gauss_params = curve_fit(gaussian_fit_func, x_data, y_lor_noise)[0]
except RuntimeError:
    fitted_gauss_params = None; print("Fit to Gaussian failed")
if fitted_gauss_params is not None:
    y_gauss_fit = gaussian_fit_func(x_data, *fitted_gauss_params)
# Lorentzian fit
try:
    fitted_lorentz_params = curve_fit(lorentzian_fit_func, x_data, y_lor_noise)[0]
except RuntimeError:
    fitted_lorentz_params = None; print("Fit to Lorentzian failed")
if fitted_lorentz_params is not None:
    y_lorentz_fit = lorentzian_fit_func(x_data, *fitted_lorentz_params)

# %% Sample data generation (line with noise)
y_line = line_func(x_data, k=0.75, b=2.5)
y_line_noise = add_noise_each_point(y_line, percentage=40)
# Parabola fit
try:
    fitted_parabola_line = curve_fit(parabola_fit_func, x_data, y_line_noise)[0]
except RuntimeError:
    fitted_parabola_line = None; print("Fit to Parabola for line failed")
if fitted_parabola_line is not None:
    y_parabola_fit_line = parabola_fit_func(x_data, *fitted_parabola_line)
# Gaussian fit
try:
    fitted_gauss_line = curve_fit(gaussian_fit_func, x_data, y_line_noise)[0]
except RuntimeError:
    fitted_gauss_line = None; print("Fit to Gaussian for line failed")
if fitted_gauss_params is not None:
    y_gauss_fit_line = gaussian_fit_func(x_data, *fitted_gauss_line)
# Lorentzian fit
try:
    fitted_lorentz_line = curve_fit(lorentzian_fit_func, x_data, y_line_noise)[0]
except RuntimeError:
    fitted_lorentz_line = None; print("Fit to Lorentzian for line failed")
if fitted_lorentz_line is not None:
    y_lorentz_fit_line = lorentzian_fit_func(x_data, *fitted_lorentz_line)

# %% Plotting results
plt.close('all')
test_lorentzian_fit = False  # test assumption that the STD for curve used for data generation is the smallest
test_line_fit = True  # test stability of fitting to the wrong type of curve (not with the bell shape)

if test_lorentzian_fit:
    plt.figure("Initial Data (Lorentzian)"); plt.plot(x_data, y_lor, 'b-', linewidth=3, label="Initial Values")
    plt.plot(x_data, y_lor_noise, 'ro', label="Values + Noise")
    plt.grid(); plt.legend(); plt.tight_layout()

    plt.figure("Fitting Results for Lorentzian")
    if fitted_parabola_params is not None:
        plt.plot(x_data, y_parabola_fit, 'g-', linewidth=3, label="Parabola Fit")
    if fitted_gauss_params is not None:
        plt.plot(x_data, y_gauss_fit, 'm-', linewidth=3, label="Gaussian Fit")
    if fitted_lorentz_params is not None:
        plt.plot(x_data, y_lorentz_fit, 'b-', linewidth=3, label="Lorentzian Fit")
    plt.plot(x_data, y_lor_noise, 'ro', label="Values + Noise")
    plt.grid(); plt.legend(); plt.tight_layout()
    if fitted_parabola_params is not None and fitted_gauss_params is not None and fitted_lorentz_params is not None:
        std_parabola_fit = np.round(np.std(y_lor_noise-y_parabola_fit), 6)
        std_gauss_fit = np.round(np.std(y_lor_noise-y_gauss_fit), 6)
        std_lorentz_fit = np.round(np.std(y_lor_noise-y_lorentz_fit), 6)

if test_line_fit:
    plt.figure("Initial Data (Line)"); plt.plot(x_data, y_line, 'b-', linewidth=3, label="Initial Values")
    plt.plot(x_data, y_line_noise, 'ro', label="Values + Noise")
    plt.grid(); plt.legend(); plt.tight_layout()

    plt.figure("Fitting Results for Line")
    if fitted_parabola_line is not None:
        plt.plot(x_data, y_parabola_fit_line, 'g-', linewidth=3, label="Parabola Fit")
    if fitted_gauss_params is not None:
        plt.plot(x_data, y_gauss_fit_line, 'm-', linewidth=3, label="Gaussian Fit")
    if fitted_lorentz_line is not None:
        plt.plot(x_data, y_lorentz_fit_line, 'b-', linewidth=3, label="Lorentzian Fit")
    plt.plot(x_data, y_line_noise, 'ro', label="Values + Noise")
    plt.grid(); plt.legend(); plt.tight_layout()
    if fitted_parabola_line is not None:
        std_parabola_fit_line = np.round(np.std(y_line_noise-y_parabola_fit_line), 6)
    if fitted_gauss_params is not None:
        std_gauss_fit_line = np.round(np.std(y_line_noise-y_gauss_fit_line), 6)
    if fitted_lorentz_line is not None:
        std_lorentz_fit_line = np.round(np.std(y_line_noise-y_lorentz_fit_line), 6)
