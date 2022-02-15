# -*- coding: utf-8 -*-
"""
Calculation of Zernike polynomials.

@author: ssklykov
"""
# %% Imports and globals
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
plt.close('all')  # closing all opened and pending figures


# %% Functions definitions
def normalization_factor(m: int, n: int) -> float:
    """
    Calculate according to the paper Honarvar, Paramersan (2013).

    Parameters
    ----------
    m : int
        Azimutal order.
    n : int
        Radial order.

    Returns
    -------
    float
        Calculated value.

    """
    n = abs(n)  # Guarantees that input parameter is always > 0
    if m == 0:
        return np.sqrt(n+1)
    else:
        return np.sqrt(2*(n+1))


def radial_polynomial(m: int, n: int, r: float) -> float:
    """
    According to the paper Honarvar, Paramersan (2013).

    Parameters
    ----------
    m: int
        Azimutal order.
    n: int
        Radial order.
    r: float
        Radius from the polar coordinate.

    Returns
    -------
    float:
        Recursively calculated radial polynomial.
    """
    if (n == 0) and (m == 0):
        return 1.0
    elif (m > n):
        return 0.0
    else:
        # Recursion formula that should be more effective than direct calculation
        return (r*(radial_polynomial(abs(m-1), n-1, r) + radial_polynomial(m+1, n-1, r)) -
                - radial_polynomial(m, n-2, r))


def triangular_function(m: int, theta: float) -> float:
    """
    Calculate triangular function according to the paper Honarvar, Paramersan (2013).

    Parameters
    ----------
    m : int
        Azimutal order.
    theta : float
        Angle from polar coordinates associated with the unit circle .

    Returns
    -------
    float
        Calculated value.

    """
    if (m > 0):
        return math.cos(m*theta)
    elif (m < 0):
        return -math.sin(m*theta)
    else:
        return 1.0


def zernike_polynomial(m: int, n: int, r: float, theta: float) -> float:
    """
    Calculate according to the paper Honarvar, Paramersan (2013).

    Parameters
    ----------
    m: int
        Azimutal order.
    n: int
        Radial order.
    r: float
        Radius from a polar coordinate system.
    theta: float
        Angle from a polar coordinate system.

    Returns
    -------
    float:
        Recursively calculated Zernike polynomial.
    """
    return (normalization_factor(m, n)*radial_polynomial(m, n, r)*triangular_function(m, theta))


def zernike_polynomials_sum(orders: list, r: float, theta: float) -> float:
    """
    Return sum of specified Zernike polynomials (m,n) for the particular point on the unit circle.

    Parameters
    ----------
    orders : list
        List of tuples like [(m,n), ...] with all Zernike polynomials orders for summing.
    r : float
        Radius on the unit circle.
    theta : float
        Angle on the unit circle.

    Raises
    ------
    TypeError
        If the input list doesn't contain tuples formatted as (m,n).

    Returns
    -------
    float
        Sum of all specified Zernike polynomials.
    """
    s = 0.0  # initial sum
    for tuple_orders in orders:
        if not (isinstance(tuple_orders, tuple)):
            raise TypeError
            break
        else:
            (m, n) = tuple_orders
            s += zernike_polynomial(m, n, r, theta)
    return s


def plot_zps_polar(orders: list, step_r: float = 0.01, step_theta: float = 1.0,
                   title: str = "Sum of few Zernike polynomials"):
    """
    Plot Zernike's polynomials sum ("zps") in polar projection for the unit radius circle.

    Parameters
    ----------
    orders : list
        List of Zernike polynomials orders recorded in tuples (m, n) inside the list like [(m, n), ...].
    step_r : float, optional
        Step for calculation of radius for a summing map (colormap). The default is 0.01.
    step_theta : float, optional
        Step (in grades) for calculation of angle for a summing map (colormap). The default is 1.0.
    title : str, optional
        Title for placing on the plot, e.g. for specific single polynomial like X Tilt. The default is "Sum of few Zernike polynomials".

    Returns
    -------
    None. The plot is shown in the separate window.

    """
    R = np.arange(0.0, 1.0+step_r, step_r)
    Theta = np.arange(0.0, 2*np.pi+np.radians(step_theta), np.radians(step_theta))
    (i_size, j_size) = (np.size(R, 0), np.size(Theta, 0))
    Z = np.zeros((i_size, j_size), dtype='float')
    for i in range(i_size):
        for j in range(j_size):
            Z[i, j] = zernike_polynomials_sum(orders, R[i], Theta[j])
    # Plotting and formatting - Polar projection + plotting the colormap as filled contour using "contourf"
    plt.figure()
    plt.axes(projection='polar')
    n_used_tones = 100
    plt.contourf(Theta, R, Z, n_used_tones, cmap=cm.coolwarm)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()


def plot_zps_rectangular(orders: list, step_xy: float = 0.02):
    """
    Plot the rectangular projection of calculated Zernike polynomials sum.

    Parameters
    ----------
    orders : list
        List of Zernike polynomials orders recorded in tuples (m, n) inside the list like [(m, n), ...].
    step_xy : float, optional
        Step in X,Y axes calculated for the range [-1, 1]. The default is 0.02.

    Returns
    -------
    None. The plot is shown in the separate window

    """
    X = np.arange(-1.0, 1.0, step_xy)
    Y = np.arange(-1.0, 1.0, step_xy)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(np.power(X, 2) + np.power(Y, 2))
    Theta = np.arctan(Y/X)
    (i_size, j_size) = (np.size(X, 0), np.size(X, 1))
    Z = np.zeros((i_size, j_size), dtype='float')
    for i in range(i_size):
        for j in range(j_size):
            Z[i, j] = zernike_polynomials_sum(orders, R[i, j], Theta[i, j])
    # Plotting and formatting - Rectangular projection
    plt.imshow(Z, cmap=cm.coolwarm)  # Plot colormap surface
    plt.axis('off')
    plt.tight_layout()


def radial_polynomial_derivative_dr(m: int, n: int, r: float) -> float:
    """
    Calculate derivative on radius of radial polynomial (like dR(m,n)/dr), that is calculated by the recurrence equation.

    Parameters
    ----------
    m: int
        Azimutal order.
    n: int
        Radial order.
    r: float
        Radius from the polar coordinate.

    Returns
    -------
    float:
        Recursively calculated derivative on radius from the radial polynomial.

    """
    if (n == 0) and (m == 0):
        return 1.0
    elif (m > n):
        return 0.0
    else:
        # Recursion formula that should be more effective than direct calculation
        return (radial_polynomial(abs(m-1), n-1, r) + radial_polynomial(m+1, n-1, r) -
                - radial_polynomial_derivative_dr(m, n-2, r))


def triangular_derivative_dtheta(m: int, theta: float) -> float:
    """
    Calculate derivative of triangular function on the angle theta (like dTriangular/dtheta).

    Parameters
    ----------
    m : int
        Azimutal order.
    theta : float
        Angle from polar coordinates associated with the unit circle .

    Returns
    -------
    float
        Calculated value.

    """
    if (m > 0):
        return -m*math.sin(m*theta)
    elif (m < 0):
        return -m*math.cos(m*theta)
    else:
        return 1.0


# %% Tests
if __name__ == '__main__':
    r = 2.0; m = -2; n = 2; theta = 0.0
    # print(normalization_factor(0,0))
    print("radial polynomial:", radial_polynomial(m, n, r))
    print("derivative of the radial polynomial: ", radial_polynomial_derivative_dr(m, n, r))
    print("zernike polynomial:", zernike_polynomial(m, n, r, theta))
    # print("sum of two polynomials 1, 1 and 0, 2:", zernike_polynomials_sum([(1, 1), (0, 2)], r, theta))

    # %% Plotting results over the surface
    # Plotting as the radial surface - Y tilt and Defocus
    step_r = 0.005
    step_theta = 1  # in grads
    orders = [(-1, 1)]  # Y tilt
    plot_zps_polar(orders, step_r, step_theta, "Y tilt")
    orders = [(-2, 2)]  # Y tilt
    plot_zps_polar(orders, step_r, step_theta, "Oblique astigmatism")
    orders = [(0, 2)]  # Defocus
    plot_zps_polar(orders, step_r, step_theta, "Defocus")
