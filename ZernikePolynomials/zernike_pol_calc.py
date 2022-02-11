# -*- coding: utf-8 -*-
"""
Calculation of Zernike polynomials.

@author: ssklykov
"""
# %% Imports and globals
import numpy as np
import math
iteration = 0


# %% Calculation functions definitions
def normalization_factor(m: int, n: int) -> float:
    """According to the paper Honarvar, Paramersan (2013)."""
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
    global iteration  # for accounting number of iterations
    if (iteration == 0):  # for starting calculation make coefficients > 0
        m = abs(m)
        n = abs(n)
        iteration += 1
    # iteration += 1  # Debugging
    # print("iteration:", iteration)  # Debugging
    # print("m:", m, "   ", "n:", n) # Debugging
    if (n == 0) and (m == 0):
        return 1.0
    elif (m > n):
        return 0.0
    else:
        # Recursion formula that should be more effective than direct calculation
        return (r*(radial_polynomial(abs(m-1), n-1, r) + radial_polynomial(m+1, n-1, r)) -
                radial_polynomial(m, n-2, r))


def triangular_function(m: int, theta: float) -> float:
    """According to the paper Honarvar, Paramersan (2013)."""
    m = abs(m)
    if (m > 0):
        return math.cos(m*theta)
    elif (m < 0):
        return -math.sin(m*theta)
    else:
        return 1.0


def zernike_polynomial(m: int, n: int, r: float, theta: float) -> float:
    """
    According to the paper Honarvar, Paramersan (2013).

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


# %% Tests
if __name__ == '__main__':
    r = 1.0; m = 1; n = 1; theta = 0.0;
    # print(normalization_factor(0,0))
    print("radial polynomial:", radial_polynomial(m, n, r))
    print("zernike polynomial:", zernike_polynomial(m, n, r, theta))
