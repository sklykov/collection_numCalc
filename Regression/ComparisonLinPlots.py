# -*- coding: utf-8 -*-
"""
Plotting the input values with error bars (an array) nad regressed values
x - values of x from initial measurements, yMean - measurements, yStD - standard deviations (errors)
@author: ssklykov
"""
import matplotlib.pyplot as plt


def PlotWErrTwo(x, yMean, yStD, xRegressed, yRegressed, title: str):
    plt.rc('font', family='serif')  # Trying to use open-source font
    plt.figure(); axes = plt.axes(); plt.title(title)
    if (min(yMean) > min(yRegressed)):
        minY = min(yRegressed)
    else:
        minY = min(yMean)
    if (max(yMean) < max(yRegressed)):
        maxY = max(yRegressed)
    else:
        maxY = max(yMean)
    if (min(x) == 0):
        xlimMin = -x[1]
    else:
        xlimMin = min(x)*0.9
    if minY == 0:
        minY = -0.2
    axes.set(xlim=(xlimMin, max(x)*1.1), ylim=(minY*0.8, maxY*1.18))  # set limits for axes
    axes.errorbar(x, yMean, yStD, fmt='ro', capsize=3, label="Original values")  # capsize - size of endings in error bars
    axes.plot(xRegressed, yRegressed, 'b-', linewidth=2, label="Regressed")
    axes.legend()  # Necessary for displaying the legend
    plt.grid()
