# -*- coding: utf-8 -*-
"""
Numerical integration - Simpson's Rule (3 points quadrature equation)
Roughly the error of integration can be estimated as ~ h**4  (the global error)
@author: ssklykov
"""
# %% Import Section
import numpy as np
from sampleFunctions import SampleFuncIntegr as sfint
import inspect
import matplotlib.pyplot as plt


# %% Algorithm implementation
def SimpsonIntegr(a: float, b: float, h: float, y, nDigits: int = 3):
    """
    Simpson's Rule Integration implementation
    Input
    -----
    a, b:
        float [a,b] interval for integration
    h:
        float, step size
    y:
        float, function or method returning single float number and accepting single float number
    """

    if (a >= b) and (int((b-a)/h) <= 1):
        print("Incosistent interval assigning [a,b] or step size h - a and b exchanged, h - decreased")
        holder = b; b = a; a = holder
        h /= 2
    if not((inspect.isfunction(y)) or (inspect.ismethod(y))):
        print("Passed function y(x) isn't the defined method or function")
        return None  # returning null object instead of any result, even equal to zero
    else:
        nPoints = int((b-a)/h) + 1
        evenSum = 0.0; oddSum = 0.0; intSum = 0.0
        for i in range(2, nPoints-2, 2):
            x = a + i*h; evenSum += y(x)
        for i in range(1, nPoints-1, 2):
            x = a + i*h; oddSum += y(x)
        intSum = (h/3)*(y(a) + y(b) + 2*evenSum + 4*oddSum)
        return round(intSum, nDigits)


# %% Adaptive calling of the Simpson's rule(above)
def AdaptiveSimpsonInt(a: float, b: float, h: float, y, nDigits: int = 3, epsilon: float = 0.01, nMaxIterations: int = 3):
    """
    Adaptive calling of Simpson's Rule for numerical integration. Epsilon - difference of two sub
    sequent calculated integrals (condition for stopping) - absolute error; nMaxIterations - maximum number of iterations
    of lowering step size h (no more than 30)
    """
    intSum1 = SimpsonIntegr(a, b, h, y, nDigits*4); h = h/2
    intSum2 = SimpsonIntegr(a, b, h, y, nDigits*4)  # nDigits should be more  than digits in epsilon!
    # As described, max int number: 2**32 -1, so it's impossible to make more iterations using nPoints evaluation
    if (nMaxIterations > 30):
        print("impossible to make so many halving iterations")
        nMaxIterations = 30
    j = 1  # number of iterations
    while((j < nMaxIterations) and (abs(intSum2-intSum1) > epsilon*intSum2)):  # |I2-I1| <= epsilon*I2 - relative error check
        intSum1 = intSum2
        h = h/2
        intSum2 = SimpsonIntegr(a, b, h, y, nDigits*4)  # nDigits should be more  than digits in epsilon!
        j += 1
    print(j, "number of iterations")
    return round(intSum2, nDigits)

# %% Parameters for testing
nDigits = 2; a = 0; b = 2; nSample = 1; h = 0.05
fClass = sfint(nDigits, nSample)  # making sample of the class contained the sample function
# making x and y values for plotting
nDigits2 = 3; a2 = 0; b2 = 2; h2 = 0.3; epsilon = 1e-4; nMaxIterations = 10
fClass2 = sfint(nDigits2, nSample)  # using the sample function with a unknown analytical integral form
nPoints2 = int(50*(b2-a2)/h2) + 1
x = np.zeros(nPoints2); y = np.zeros(nPoints2)
for i in range(nPoints2):
    x[i] = a + i*(h2/50); y[i] = fClass2.sampleF(x[i])
# plot sample function from interval [a,b]
fig = plt.figure(); plt.plot(x,y); plt.grid()

# %% Testing
integral = SimpsonIntegr(a, b, h, fClass.sampleF, nDigits)
print(integral, " - calculated integral value for 1st sample f(x)")
print("1 - exact value from Newton-Leibniz equation F(b) - F(a)")  # F(a) = 0; F(b) = 1; F(x) = x^3 - x^2 - 1.5*x
integral2 = SimpsonIntegr(a2, b2, h2, fClass2.sampleF, nDigits2)
print(integral2, " - integral value for 1st sample f(x) w/t adaptation")
integral22 = AdaptiveSimpsonInt(a2, b2, h2, fClass2.sampleF, nDigits2, epsilon, nMaxIterations)
print(integral22, " - integral value for 2nd sample f(x) with adaptation")
