# -*- coding: utf-8 -*-
"""
Numerical Integration testing - Trapezoidal rule with constant step h for creating mesh points.

Trapezoidal rule integration precision can be roughly estimated as ~ h**2.

@author: sklykov
@license: The Unlicense
"""
# %% Import Section
import numpy as np
from sampleFunctions import SampleFuncIntegr as sfint
import inspect
import matplotlib.pyplot as plt


# %% Algorithm implementation
def TrapezoidalIntegr(a: float, b: float, h: float, y, nDigits: int = 3) -> float:
    """
    Trapezoidal Integration implementation.

    It demands [a,b] of interval for integration; h - step size, y - function or
    method returning single float number and accepting single float number.

    Parameters
    ----------
    a : float
        Lower bond of integration interval.
    b : float
        Higher bond of integration interval.
    h : float
        Integration step, should be less than (b-a).
    y : callable function / method
        Function for which integration undergoes.
    nDigits : int, optional
        Number of returning digits after float point. The default is 3.

    Returns
    -------
    float
        Integral sum.

    """
    if (a >= b) and (int((b-a)/h) <= 1):
        print("Incosistent interval assigning [a,b] or step size h")
        return None  # returning null object instead of any result, even equal to zero
    elif not ((inspect.isfunction(y)) or (inspect.ismethod(y))):
        print("Passed function y(x) isn't the defined method or function")
        return None
    else:
        nPoints = int((b-a)/h) + 1
        intSum = 0.0
        for i in range(1, nPoints-1):
            x = a + i*h; intSum += y(x)
        intSum += (y(a) + y(b))/2
        intSum *= h
        return round(intSum, nDigits)


# %% Parameters for testing
nDigits = 2; a = 0; b = 2; nSample = 1; h = 0.05
nPoints = int((b-a)/h) + 1
# print(nPoints," - nPoints inside the interval [a,b]")
x = np.zeros(nPoints); y = np.zeros(nPoints)
fClass = sfint(nDigits, nSample)  # making sample of the class contained the sample function
# making x and y values for plotting
for i in range(nPoints):
    x[i] = a + i*h; y[i] = fClass.sampleF(x[i])

# plot sample function from interval [a,b]
plt.close('all')
fig = plt.figure(); plt.plot(x, y); plt.grid()

# %% Testing
integral = TrapezoidalIntegr(a, b, h, fClass.sampleF, nDigits)
print(integral, " - calculated integral value")
print("1 - exact value from Newton-Leibniz equation F(b) - F(a)")  # F(a) = 0; F(b) = 1; F(x) = x^3 - x^2 - 1.5*x
