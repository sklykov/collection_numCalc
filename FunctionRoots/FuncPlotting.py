# -*- coding: utf-8 -*-
"""
Make the plotting a bit more automated by using two function containing all needed function calls.

@author: sklykov
@license: The Unlicense
"""
# %% Import section
from matplotlib import pyplot as plt
import numpy as np


# %% Function for plotting input function returning real values
def plotMyExampleFunc(func, nExample, a: float, b: float, nPoints: int):
    """
    Plotting input EXAMPLE function in the range [a,b] with nPoints equally aligned in the specified interval.
    """
    x = np.linspace(a, b, nPoints)  # equally disturbed x values
    y = np.zeros(nPoints, dtype=float)  # for holding y values
    x0 = np.zeros(nPoints, dtype=float)  # For marking the zero X axis
    for i in range(nPoints):
        y[i] = func(x[i], nExample)  # It's implied that function returning the float numbers
    # Setting up the graphic properties. More details - in relevant section of this repository
    plt.rc('font', family='serif')  # Trying to use open-source font
    plt.figure()
    axes = plt.axes()
    axes.plot(x, y, 'r-', linewidth=3)  # The main plotting function - plotting of algebraic function of interest
    axes.plot(x, x0, 'b-', linewidth=3)  # Plotting the X axis
    axes.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)))  # set limits for axes
    plt.grid()


# %% Function for plotting any input function
def plotMyAwesomeFunc(func, a: float, b: float, nPoints: int):
    """
    Plotting any (in theory) input function (returning the float number) in the range [a,b] with
    nPoints equally aligned in the specified interval.
    """
    x = np.linspace(a, b, nPoints)  # equally disturbed x values
    y = np.zeros(nPoints, dtype=float)  # for holding y values
    x0 = np.zeros(nPoints, dtype=float)  # For marking the zero X axis
    for i in range(nPoints):
        y[i] = func(x[i])  # It's implied that function returning the float numbers
    # Setting up the graphic properties. More details - in relevant section of this repository
    plt.rc('font', family='serif')  # Trying to use open-source font
    plt.figure()
    axes = plt.axes()
    axes.plot(x, y, 'r-', linewidth=3)  # The main plotting function - plotting of algebraic function of interest
    axes.plot(x, x0, 'b-', linewidth=3)  # Plotting the X axis
    axes.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)))  # set limits for axes
    plt.grid()
