# -*- coding: utf-8 -*-
"""
Implementation of simple Euler solver and Euler solver with correction for 1D simple ODE like: y'(x) = f(x).

@author: sklykov
@license: The Unlicense
"""
# %% Import section
import SampleF  # (!): import module allows to import docstrings as well
import inspect
import numpy as np
import matplotlib.pyplot as plt


# %% The simple Euler solver
def simpleEuler(f, h: float, y0: float, xStart: float, xFinish: float) -> tuple:
    """
    Implement simple single step (~h) ODE solver.

    f - input function (y'(x) = f(x)), h - step size, y0 - initial value,
    xStart - first value for solving an input ODE, xFinish - finishing value for calculations.
    Returning a tuple that composes of xMesh values and yValues - the solution for an ODE
    """
    # Checking input parameters for validity
    if (checkInputs(f, h, xStart, xFinish)):
        return None
    # Defying exact number of steps between interval points [xStart,xFinish]
    # Defying exact number of steps between interval points [xStart,xFinish] or number of points for modelling and returning
    n = numberOfSteps(xStart, xFinish, h); nEven = int((xFinish-xStart) / h) + 1
    yValues = np.zeros(n, dtype='float'); xMesh = np.zeros(n, dtype='float')  # init mesh values
    yValues[0] = float(y0); xMesh[0] = float(xStart)
    # Calculations
    if (n == nEven):
        for i in range(1, n):
            yValues[i] = yValues[i-1] + h*f(xMesh[i-1])  # simple approximation
            xMesh[i] = xMesh[i-1] + h
    else:
        for i in range(1, n-1):
            yValues[i] = yValues[i-1] + h*f(xMesh[i-1])  # simple approximation
            xMesh[i] = xMesh[i-1] + h
        hLastStep = xFinish - xMesh[n-2]
        xMesh[n-1] = xFinish; yValues[n-1] = yValues[n-2] + hLastStep*f(xMesh[n-1])

    return (xMesh, yValues)


# %% The Euler solver with predictor - corrector
def predcorrEuler(f, h: float, y0: float, xStart: float, xFinish: float) -> tuple:
    """
    Implement simple single step (~h*2) ODE solver with prediction / correction additional step.

    f - input function (y'(x) = f(x)),
    h - step size, y0 - initial value, xStart - first value for solving an input ODE, xFinish - finishing value for
    calculations. Returning a tuple that composes of xMesh values and yValues - the solution for an ODE
    """
    # Checking input parameters for validity
    if (checkInputs(f, h, xStart, xFinish)):
        return None
    # Defying exact number of steps between interval points [xStart,xFinish] or number of points for modelling and returning
    n = numberOfSteps(xStart, xFinish, h); nEven = int((xFinish-xStart) / h) + 1
    # Values initialization
    yValues = np.zeros(n, dtype='float'); xMesh = np.zeros(n, dtype='float')  # init mesh values
    yValues[0] = float(y0); xMesh[0] = float(xStart)
    # Calculations
    if (n == nEven):
        for i in range(1, n):
            # yTemp = yValues[i-1] + h*f(xMesh[i-1]) # prediction value - temporary one (now - not necessary because f(x) only)
            xMesh[i] = xMesh[i-1] + h
            yValues[i] = yValues[i-1] + 0.5*h*(f(xMesh[i-1]) + f(xMesh[i]))  # prediction - correction values
    else:
        for i in range(1, n-1):
            # yTemp = yValues[i-1] + h*f(xMesh[i-1]) # prediction value - temporary one (now - not necessary because f(x) only)
            xMesh[i] = xMesh[i-1] + h
            yValues[i] = yValues[i-1] + 0.5*h*(f(xMesh[i-1]) + f(xMesh[i]))  # prediction - correction values
        hLastStep = xFinish - xMesh[n-2]
        xMesh[n-1] = xFinish; yValues[n-1] = yValues[n-2] + 0.5*hLastStep*(f(xMesh[n-2]) + f(xMesh[n-1]))
    return (xMesh, yValues)


# %% Checking input parameters function
def checkInputs(f, h: float, xStart: float, xFinish: float) -> bool:
    if not (inspect.ismethod(f) or inspect.isfunction(f)):
        print("Specified as function 'f' isn't a callable function or method")
        return True
    elif (xFinish <= xStart):
        print("This implementation supposes that xFinish point > xStart")
        return True
    elif (h <= 0):
        print("This implementation supposes that h > 0")
        return True
    else:
        return False


# %% Calculation of step number
def numberOfSteps(xStart: float, xFinish: float, h: float) -> int:
    n = 0  # initial number of steps
    nEven = int((xFinish-xStart) / h) + 1
    xAcc = xStart
    for i in range(nEven):
        xAcc += h
    epsilon = 1e-6  # for measuring of equality of two float numbers
    if ((abs(xFinish - xAcc) < epsilon) or (xFinish == xAcc)):  # comparison of two floats numbers with some precision (epsilon)
        n = nEven
    else:
        n = nEven + 1
    return n


# %% Testing values
def main():
    h = 0.25; y0 = 0.0; xStart = 0.0; xFinish = 3.0; nDigits = 2
    # For referring to a function it can be called without () notation
    f1 = SampleF.SampleFunctions(1, nDigits); y1 = f1.getValue
    # print(y1.__doc__) # Check importing of docstrings as well (as part of a module documentation)
    (x, y) = simpleEuler(y1, h, y0, xStart, xFinish)
    (x2, y2) = predcorrEuler(y1, h, y0, xStart, xFinish)
    # %% Theoretical values calculation
    yTh = np.zeros(len(x), dtype='float')
    for i in range(len(x)):
        yTh[i] = f1.getTheoreticalValue(x[i])
    yTh2 = np.zeros(len(x2), dtype='float')
    for i in range(len(x2)):
        yTh2[i] = f1.getTheoreticalValue(x2[i])
    # %% Testing results - comparison on graphs
    # Making 1st graph
    x.round(nDigits); y.round(nDigits)
    plt.figure(); plt.xlabel("x"); plt.ylabel("y(x) solution"); plt.plot(x, y, label='numerical solution', linewidth=3)
    plt.plot(x, yTh, 'ro-', label='theoretical function', linewidth=1); plt.legend(); plt.title("Simple Euler solver")
    # Making 2nd graph
    x2.round(nDigits); y2.round(nDigits)
    plt.figure(); plt.xlabel("x"); plt.ylabel("y(x) solution"); plt.plot(x2, y2, label='numerical solution', linewidth=3)
    plt.plot(x2, yTh2, 'ro-', label='theoretical function', linewidth=1)
    plt.legend(); plt.title("Euler solver with prediction / correction")


# %% For preventing execution test functions above while import to another modules
if __name__ == "__main__":
    main()
