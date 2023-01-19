# -*- coding: utf-8 -*-
"""
Numerical integration - trapezoidal rule with adaptive selection of integration step h.

This adaptive behaviour is intended for calculation integral value with epsilon precision (making step less (h/2) doesn't
increase the precision of calculation).

@author: sklykov
@license: The Unlicense
"""
# %% Import Section
import numpy as np
from sampleFunctions import SampleFuncIntegr as sfint
import inspect
import matplotlib.pyplot as plt


# %% Algorithm implementation
def TrapezoidalAdaptiveInt(a: float, b: float, h: float, y, nDigits: int = 3,
                           epsilon: float = 0.01, nMaxIterations: int = 10) -> float:
    """
    Trapezoidal Integration implementation.

    It demands [a,b] of interval for integration; h - step size, y - function or
    method returning single float number and accepting single float number; epsilon - difference of two sub
    sequent calculated integrals (condition for stopping) - absolute error; nMaxIterations - maximum number of iterations
    of lowering step size h (no more than 30).

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
    epsilon : float, optional
        Difference between integration sum that is considered to be small enough. The default is 0.01.
    nMaxIterations : int, optional
        Number of maximum iterations for finding the optimal integral step. The default is 10. The maximum is 30.

    Returns
    -------
    float
        DESCRIPTION.

    """
    if (a >= b) and (int((b-a)/h) <= 1):
        print("Incosistent interval assigning [a,b] or step size h")
        return None  # returning null object instead of any result, even equal to zero
    elif not ((inspect.isfunction(y)) or (inspect.ismethod(y))):
        print("Passed function y(x) isn't the defined method or function")
        return None
    else:
        nPoints = int((b-a)/h) + 1
        intSum1 = 0.0
        for i in range(1, nPoints-1):
            x = a + i*h; intSum1 += y(x)
        intSum1 += (y(a) + y(b))/2
        intSum1 *= h
        h = h/2
        nPoints = int((b-a)/h) + 1
        intSum2 = 0.0
        for i in range(1, nPoints-1):
            x = a + i*h; intSum2 += y(x)
        intSum2 += (y(a) + y(b))/2
        intSum2 *= h
        # As described, max int number: 2**31 -1, so it's impossible to make more iterations using nPoints evaluation
        if (nMaxIterations > 30):
            print("impossible to make so many halving iterations")
            nMaxIterations = 30
        j = 1  # number of iterations for obtaining
        # |I2-I1| <= epsilon*I2 - relative error
        while ((j < nMaxIterations) and (abs(intSum1-intSum2) > epsilon*intSum2)):
            intSum1 = intSum2
            h = h/2
            nPoints = int((b-a)/h) + 1
            intSum2 = 0.0
            for i in range(1, nPoints-1):
                x = a + i*h; intSum2 += y(x)
            intSum2 += (y(a) + y(b))/2
            intSum2 *= h
            j += 1
        intSum = intSum2
        print(j, "- number of iterations made")
        return round(intSum, nDigits)


# %% Parameters for testing
nDigits = 2; a = 0; b = 2; nSample = 1; h = 0.5; nMaxIterations = 10; epsilon = 0.01
hGraph = h/50
nPoints = int((b-a)/hGraph) + 1
x = np.zeros(nPoints); y = np.zeros(nPoints)
fClass = sfint(nDigits, nSample)  # making sample of the class contained the sample function
# making x and y values for plotting
for i in range(nPoints):
    x[i] = a + i*hGraph; y[i] = fClass.sampleF(x[i])

# plot sample function from interval [a,b]
plt.close("all")
fig = plt.figure(); plt.plot(x, y); plt.grid()

# %% Testing
integral = TrapezoidalAdaptiveInt(a, b, h, fClass.sampleF, nDigits, epsilon, nMaxIterations)
print(integral, " - calculated integral value")
print("1 - exact value from Newton-Leibniz equation F(b) - F(a)")  # F(a) = 0; F(b) = 1; F(x) = x^3 - x^2 - 1.5*x
