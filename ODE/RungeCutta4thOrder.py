# -*- coding: utf-8 -*-
"""
Implementation of 4th order Runge - Kutta method for the ODE solving like y'(x) = f(x)
The implemented equation is analogue to the Simpon's integration rule
@author: ssklykov
"""
# %% Import section
import SampleF
import numpy as np
import matplotlib.pyplot as plt
from EulerSolver import predcorrEuler, numberOfSteps, checkInputs

# %% Implementation of the algorithm
def rungeCutta4(f,h:float,y0:float,xStart:float,xFinish:float) -> tuple:
    """Simple single step (~h*4) ODE Runge-Cutta solver. f - input function (y'(x) = f(x)),\
        h - step size, y0 - initial value, xStart - first value for solving an input ODE, xFinish - finishing value for \
        calculations. Returning a tuple that composes of xMesh values and yValues - the solution for an ODE"""
    # Checking input parameters for validity
    if (checkInputs(f, h, xStart, xFinish)):
        return None
    # Defying exact number of steps between interval points [xStart,xFinish] or number of points for modelling and returning
    n = numberOfSteps(xStart, xFinish, h); nEven = int((xFinish-xStart) / h) + 1
    # Values initialization
    yValues = np.zeros(n,dtype='float'); xMesh = np.zeros(n,dtype='float') # init mesh values
    yValues[0] = float(y0); xMesh[0] = float(xStart)
    # Calculations
    if (n == nEven):
        for i in range(1,n):
            xMesh[i] = xMesh[i-1] + h
            yValues[i] = yValues[i-1] + (h*(f(xMesh[i-1]) + f(xMesh[i]) + 4*f(xMesh[i-1]+0.5*h))/6) # like Simpson's rule
    else:
        for i in range(1,n-1):
            xMesh[i] = xMesh[i-1] + h
            yValues[i] = yValues[i-1] + (h*(f(xMesh[i-1]) + f(xMesh[i]) + 4*f(xMesh[i-1]+0.5*h))/6)
        hLastStep = xFinish - xMesh[n-2]; xMesh[n-1] = xFinish
        yValues[n-1] = yValues[n-2] + (hLastStep*(f(xMesh[n-2]) + f(xMesh[n-1]) + 4*f(xMesh[n-2]+0.5*h))/6)
    return (xMesh,yValues)

# %% Testing parameters
h = 0.25; y0 = 0.0; xStart = 0.0; xFinish = 4.0; nDigits = 2
#%% Testing
f1 = SampleF.SampleFunctions(1,nDigits); y1 = f1.getValue # For referring to a function it can be called without () notation
(x,y) = predcorrEuler(y1,h,y0,xStart,xFinish)
(x2,y2) = rungeCutta4(y1,h,y0,xStart,xFinish)

# %% Theoretical values calculation
yTh = np.zeros(len(x),dtype='float')
for i in range(len(x)):
    yTh[i] = f1.getTheoreticalValue(x[i])
yTh2 = np.zeros(len(x2),dtype='float')
for i in range(len(x2)):
    yTh2[i] = f1.getTheoreticalValue(x2[i])
# %% Comparison results on grahps
# x.round(nDigits); y.round(nDigits)
# plt.figure(); plt.xlabel("x"); plt.ylabel("y(x) solution"); plt.plot(x,y,label='numerical solution',linewidth=3)
# plt.plot(x,yTh,label='theoretical function',linewidth=2); plt.legend(); plt.title("Euler solver with prediction / correction")

# %% 2nd Comparison results on graphs
x2.round(nDigits); y2.round(nDigits)
plt.figure(); plt.xlabel("x"); plt.ylabel("y(x) solution"); plt.plot(x2,y2,'bo',label='numerical solution',linewidth=4)
plt.plot(x2,yTh2,'r',label='theoretical function',linewidth=2); plt.legend(); plt.title("Runge-Cutta 4th order solver")