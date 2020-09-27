# -*- coding: utf-8 -*-
"""
Non-comprehensive implementation of Numerov's method for solving ODEs like y'' = f(x)*y + g(x)
Applicability -? and Testing results -?
@author: ssklykov
"""
# %% Import section
import numpy as np
import math
from EulerSolver import checkInputs, numberOfSteps


# %% Sample functions definition here - just for quick testing - some crazy functions
def sampleFunc1(x: float) -> float:
    f = math.pow(x, 2)
    return f


def sampleFunc2(x: float) -> float:
    g = 0.1*math.sqrt(x)
    return g


# %% Numerov's method implementation
def numerovMethod(y0: float, y1: float, f, g, h: float, xStart: float, xFinish: float) -> tuple:
    """This method required for calculation of value y[m+1] already precaclculated values y[m-1] and y[m] \
        This method is suitable for solution of equations y'' = f(x)*y + g(x) only!"""
    if checkInputs(f, h, xStart, xFinish) and checkInputs(g, h, xStart, xFinish):
        return None
    n = numberOfSteps(xStart, xFinish, h)
    yValues = np.zeros(n,dtype='float'); xMesh = np.zeros(n,dtype='float')  # init mesh values
    xMesh[0] = xStart; xMesh[1] = xStart + h
    yValues[0] = y0; yValues[1] = y1
    for i in range(2, n):
        xMesh[i] = xMesh[i-1] + h
        umCurrent = 1 - (h*h*f(xMesh[i])/12)
        um = 1 - (h*h*f(xMesh[i-1])/12)
        umPrev = 1 - (h*h*f(xMesh[i-2])/12)
        yF1 = ((12-10*um)*yValues[i-1] - umPrev*yValues[i-2])/(umCurrent)
        yF2 = (h*h*(g(xMesh[i]) + 10*g(xMesh[i-1]) + g(xMesh[i-2])))/(12*umCurrent)
        yValues[i] = yF1 + yF2
    return (xMesh, yValues)


# %% Testing values and getting the results
# TODO: make theoretically solved example (the ground truth example)
y0 = 0; y1stDeriv0 = 0; h = 0.01  # Partially - concluded from the function form
y1stDeriv1 = y1stDeriv0 + h*(sampleFunc1(h)*y0 + sampleFunc2(h))  # Approximation of next value using Euler - Cromer equation
y1 = y0 + y1stDeriv1*h  # see the previous comment
nPoints = 1000; nDigits = 3
xStart = 0; xFinish = 4

# Testing (results - ?)
(x, y) = numerovMethod(y0, y1, sampleFunc1, sampleFunc2, h, xStart, xFinish)
