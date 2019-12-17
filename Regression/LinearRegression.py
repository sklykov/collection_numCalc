# -*- coding: utf-8 -*-
"""
Linear regression using the deviated randomly linear dependency (y(x) = a*x + b)
Developed in the Spyder IDE using Kite

@author: ssklykov
"""
# %% Import section
from SampleValues import GenerateSample
# %% Controlling / modelling values
a = 2; b = 1 # From dependency y(x) = a*x + b
n = 10  # number of sampling points
nDigits = 2  # Precision of calculation / rounding
percentError = 15  # An error controlling deviations in generated values
xMin = 0; xMax = 5  # Controlling minimal and maximal values from an interval
nSamples = 11 # 10 + 1 samples

# %% Generating the sample values - from a linear dependency
values = GenerateSample(a,b,xMin,xMax,nSamples,percentError,nDigits)
(x,yMean,yStD) = values.generateSampleValues()