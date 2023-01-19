# -*- coding: utf-8 -*-
"""
Linear regression using the deviated randomly linear dependency (y(x) = a*x + b).

@author: sklykov
@license: The Unlicense
"""
# %% Import section
from SampleValues import GenerateSample
import numpy as np
from ComparisonLinPlots import PlotWErrTwo
# %% Controlling / modelling values
a = 2; b = 1  # From dependency y(x) = a*x + b
n = 10  # number of sampling points (modelling measures from some awesome experiment)
nDigits = 2  # Precision of calculation / rounding
percentError = 30  # An error controlling deviations in generated values [%]
xMin = 0; xMax = 5  # Controlling minimal and maximal values from an interval [a,b]
nSamples = 11  # 10 + 1 samples, specify always in this manner, because np.linspace(a,b,n) - including "a" from the interval

# %% Generating the sample values - from a linear dependency
values = GenerateSample(a, b, xMin, xMax, nSamples, percentError, nDigits)
(x, yMean, yStD) = values.generateSampleValues()


# %% Linear Regression
def LinearRegression(x, yMean, yStD, nDigits):
    # Interim values calculation from the book (Sx, Sxx, etc)
    S = 0.0; Sx = 0.0; Sxx = 0.0; Sy = 0.0; Sxy = 0.0; sigma2 = 0.0
    aRegressed = 0; bRegressed = 0
    for i in range(len(x)):
        sigma2 = 1 / pow(yStD[i], 2)
        S += sigma2; Sx += x[i]*sigma2; Sy += yMean[i]*sigma2
        Sxx += pow(x[i], 2)*sigma2; Sxy += (x[i]*yMean[i])*sigma2
    delta = (S*Sxx) - pow(Sx, 2)
    if (delta != 0):
        aRegressed = (S*Sxy - Sx*Sy)/delta; bRegressed = (Sy*Sxx - Sx*Sxy)/delta
        aRegressed = round(aRegressed, nDigits); bRegressed = round(bRegressed, nDigits)
    return (aRegressed, bRegressed)


# %% Regression evaluation
(aR, bR) = LinearRegression(x, yMean, yStD, nDigits)

# %% Plotting
# Generation of arrays for plotting
nRegressed = (nSamples-1)*10 + 1  # For plotting - calculation in 10 times more points between specified interval [xMin,xMax]
xRegressed = np.linspace(xMin, xMax, nRegressed)
yRegressed = np.zeros(nRegressed)
for i in range(nRegressed):
    yRegressed[i] = aR*xRegressed[i] + bR
# Plot
PlotWErrTwo(x, yMean, yStD, xRegressed, yRegressed, "Linear Regression results")
