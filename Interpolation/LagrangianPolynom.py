# -*- coding: utf-8 -*-
"""
Interpolation methods implementation
The Lagrangian polynom construction and testing
Developed in the Spyder IDE
@author: ssklykov
"""
# %% Import section
from InterpolSamples import InterpolSamples
from matplotlib import pyplot as plt
import numpy as np

# %% Testing values
nPoints = 12; nPoints += 1 # Adding 1 to the number points for making equal steps in the interval [a,b]
percentageError = 10  # For making disturbed values
startInterval = 0 # a from [a,b]
finishInterval = 10 # b from [a,b]
nDigit = 3 # as in the sample function inside the class InterpolSamples
nTimesPointsMore = 50  # how many points one want to generate more than it was in a sample (defined by nPoints)

# %% Generation sample points and plot them
intps = InterpolSamples(nPoints,percentageError,startInterval,finishInterval)
(x,y,yEr) = intps.valuesForInterpol() # Generation sample points

# Plotting
plt.rc('font',family='serif') # Trying to use open-source font
plt.figure(); axes = plt.axes(); plt.title("Sample values")
if (min(y) > min(yEr)):
    minY = min(yEr)
else:
    minY = min(y)
if (max(y) < max(yEr)):
    maxY = max(yEr)
else:
    maxY = max(y)
axes.set(xlim=(min(x),max(x)),ylim=(minY,maxY*1.1))  # set limits for axes
axes.plot(x,y,'ro-',linewidth=2,label="y w/t errors")
axes.plot(x,yEr,'bo-',linewidth=2,label="y values with errors")
axes.legend() # Necessary for displaying the legend
plt.grid()

# %% Evaluation of y values for making interpolation
def LagrangianPol(xIn:float,xArray,yArray,nRoundDigit):
    # xIn - x value for that the returning value y should be calculated
    # xArray - x values for which y values from yArray are known
    yReturn = 0
    # Outer loop - sum of polynomal coefficient * y[i] (y[i] - known values)
    for i in range(len(xArray)):
        p = 1
        # Inner loop  - for calculation of coefficients from polynomals
        for j in range(len(xArray)):
            if (i != j):
                p *= (xIn - xArray[j])/(xArray[i]-xArray[j])

        yReturn += p*yArray[i]

    return round(yReturn,nRoundDigit)

# %% Calculation of interpolated values for further comparisons by plotting
xInterpol = np.zeros((len(x)-1)*nTimesPointsMore + 1) # For these values should be the function interpolated
yInterpol = np.zeros((len(x)-1)*nTimesPointsMore + 1) # For holding interpolation values
yInterpolEr = np.zeros((len(x)-1)*nTimesPointsMore + 1) # For holding interpolation values
yTheoretical = np.zeros((len(x)-1)*nTimesPointsMore + 1) # For holding interpolation values
# Generation x values for interpolations
xInterpol[0] = x[0]
for i in range(1,len(xInterpol)):
    xInterpol[i] = xInterpol[i-1] + ((x[1]-x[0])/nTimesPointsMore)
# Get interpolated values for yPure
for i in range(len(yInterpol)):
    yInterpol[i] = LagrangianPol(xInterpol[i],x,y,nDigit)
# Get interpolated values for yEr
for i in range(len(yInterpolEr)):
    yInterpolEr[i] = LagrangianPol(xInterpol[i],x,yEr,nDigit)
# Get theoretical (tabulated) values for y(x)
for i in range(len(yTheoretical)):
    yTheoretical[i] = InterpolSamples.sampleFunction(xInterpol[i])

# %% Again plotting for making comparisons (visual)
plt.rc('font',family='serif') # Trying to use open-source font
plt.figure(); axes = plt.axes(); plt.title("pure y + interpolated values")
axes.set(xlim=(min(x),max(x)),ylim=(min(y),max(y)*1.1))  # set limits for axes
axes.plot(x,y,'ro',label="y w/t errors")
axes.plot(xInterpol,yInterpol,'g-',linewidth=3,label="y interpolated")
axes.plot(xInterpol,yTheoretical,'m-',linewidth=1,label="y theoretical")
axes.legend() # Necessary for displaying the legend
plt.grid()

# Figure 2
plt.rc('font',family='serif') # Trying to use open-source font
plt.figure(); axes = plt.axes(); plt.title("error y + interpolated values")
axes.set(xlim=(min(x),max(x)),ylim=(min(yTheoretical),max(yTheoretical)*1.1))  # set limits for axes
axes.plot(x,yEr,'bo',label="y with errors")
axes.plot(xInterpol,yInterpolEr,'g-',linewidth=2,label="y interpolated")
axes.plot(xInterpol,yTheoretical,'m-',linewidth=1,label="y theoretical")
axes.legend() # Necessary for displaying the legend
plt.grid()