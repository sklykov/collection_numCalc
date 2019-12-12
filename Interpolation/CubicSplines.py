# -*- coding: utf-8 -*-
"""
Cubic Splines implementation for finding interpolation points

Developed in the Spyder IDE
@author: ssklykov
"""
# %% Import section
from InterpolSamples import InterpolSamples
import numpy as np
from PlottingArrays import plotTwoArrDiffX as plotTwo
from PlottingArrays import plotSingleArrSameX as plotSingle
from TridiagonalSol import Solution as solve3diag

# %% Testing values
nPoints = 20; nPoints += 1 # Adding 1 to the number points for making equal steps in the interval [a,b]
percentageError = 15  # For making disturbed values
startInterval = 0 # a from [a,b]
finishInterval = 8 # b from [a,b]
nDigits = 3 # as in the sample function inside the class InterpolSamples
nTimesPointsMore = 10  # how many points one want to generate more than it was in a sample (defined by nPoints)

# %% Generation sample points and plot them
intps = InterpolSamples(nPoints,percentageError,startInterval,finishInterval)
(x,y,yEr) = intps.valuesForInterpol() # Generation sample points

# Plotting
#plotTwo(x,y,yEr,"Sample values","y w/t errors","y values with errors")
plotSingle(x,y,"Sample value w/t errors")

# %% Cubic spline coefficients calculation for equation A*x^3 + B*x^2 + C*x + D
""" Calculation of cubic spline coefficients (A,B,C,D) with free or natural boundaries condition (y''(a) = y''(b) = 0) """
def cubicSplineCoeffCalc(x,y):
    # Coefficients for calculations
    A = np.zeros(len(x),dtype='float'); B = np.zeros(len(x),dtype='float')
    C = np.zeros(len(x),dtype='float'); D = np.zeros(len(x),dtype='float')
    # interim values for coefficient calculation for free boundary conditions
    Deriv = np.zeros(len(x),dtype='float'); a = np.zeros(len(x),dtype='float'); b = np.zeros(len(x),dtype='float')
    c = np.zeros(len(x),dtype='float'); d = np.zeros(len(x),dtype='float')

    # First and last values defined explicitly (the FREE BOUNDARIES CONDITION CASE!)
    c[0] = 0; c[len(c)-1] = 0
    a[0] = 0; a[len(a)-1] = 0
    d[0] = 0; d[len(d)-1] = 0
    b[0] = 2*(x[1] - x[0]); b[len(b)-1] = 2*(x[len(x)-1]-x[len(x)-2])
    # Calculation of interim values a,b,c,d
    for i in range(1,len(x)-1):
        hI = (x[i+1]-x[i]); hIm1 = (x[i] - x[i-1])
        a[i] =  hIm1 # h[i-1]
        b[i] = 2*(hIm1 + hI)
        c[i] = hI # h[i]
        d[i] = 6*(((y[i+1]-y[i])/hI)-((y[i]-y[i-1])/hIm1))
    # Calculation of second derivatives Deriv - solution of tridiagonal linear system of equations Gx = D,
    # there G composes of a,b,c,d
    Deriv = solve3diag(a,b,c,d)

    # Calculation of coefficients for polynoms that interpolated values inside the intervals [x[i],x[i+1]]
    for i in range(0,len(x)-1):
        hI = (x[i+1]-x[i])
        A[i] = (Deriv[i+1]-Deriv[i])/(6*hI)
        B[i] = (Deriv[i]*x[i+1] - Deriv[i+1]*x[i])/(2*hI)
        C[i] = ((Deriv[i+1]*pow(x[i],2) - Deriv[i]*pow(x[i+1],2))/(2*hI)) + ((y[i+1]-y[i])/hI) - (A[i]*pow(hI,2))
        D[i] = ((Deriv[i]*pow(x[i+1],3) - Deriv[i+1]*pow(x[i],3))/(6*hI)) + ((y[i]*x[i+1]-y[i+1]*x[i])/hI) - ((B[i]*pow(hI,2))/3)
    return (A,B,C,D)

# %% Interpolation using previously calculated coefficients
""" Interpolation of input values using previous. It uses the coefficients calculated before """
def cubicInterpolation(cubicCoefficients:tuple,xCustom:float,nRoundingDigits:int):
    # Defining the index number of interval [x[i],x[i+1]]
    (A,B,C,D) = cubicCoefficients  # Really not effictive passing arguments but it's quick workaround up to now
    intervalFound = False; i = 0
    while (not intervalFound) and (i < (len(A)-1)):
        if (xCustom >= x[i]) and (xCustom <= x[i+1]):
            intervalFound = True;
        else:
            i += 1
    yVal = None
    if(intervalFound):
        yVal = A[i]*(pow(xCustom,3)) + B[i]*(pow(xCustom,2)) + C[i]*xCustom + D[i]
        yVal = round(yVal,nRoundingDigits)
    else:
        print("the requested x doesn't lay in the interval [startInterval,finishInterval]")
    return yVal

# %% Testing interpolation
coeff = cubicSplineCoeffCalc(x,y)  # Coefficients calculation for splines
xInterpol = np.zeros((len(x)-1)*nTimesPointsMore + 1) # For these values should be the function interpolated
yInterpol = np.zeros((len(x)-1)*nTimesPointsMore + 1) # For holding interpolation values
# Generation x values for interpolations
xInterpol[0] = x[0]
for i in range(1,len(xInterpol)):
    xInterpol[i] = xInterpol[i-1] + ((x[1]-x[0])/nTimesPointsMore)
# Get interpolated values for yPure
for i in range(len(yInterpol)):
    yInterpol[i] = cubicInterpolation(coeff,xInterpol[i],nDigits)

# %% Plotting results of interpolation
plotTwo(x,xInterpol,y,yInterpol,"Interpolation results","original","interpolated")
