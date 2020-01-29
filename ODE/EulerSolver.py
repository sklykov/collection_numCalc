# -*- coding: utf-8 -*-
"""
Implementation of simple Euler solver and Euler solver with correction for 1D simple ODE like: y'(x) = f(x)

@author: ssklykov
"""

#%% Import section
import SampleF # (!): import module allows to import docstrings as well
import inspect
import numpy as np
import matplotlib.pyplot as plt

#%% Top function for accessing both Euler methods
def simpleEuler(f,h:float,y0:float,xStart:float,xFinish:float) -> tuple:
    """Simple single step (~h) ODE solver. f - input function (y'(x) = f(x)), h - step size, y0 - initial value,\
        xStart - first value for solving an input ODE, xFinish - finishing value for calculations. Returning \
        a tuple that composes of xMesh values and yValues - the solution for an ODE"""
    if not(inspect.ismethod(f) or inspect.isfunction(f)):
        print("Specified as function 'f' isn't a callable function or method")
        return None
    elif (xFinish <= xStart):
        print("This implementation supposes that xFinish point > xStart")
        return None
    elif (h <= 0):
        print("This implementation supposes that h > 0")
        return None
    # Defying exact number of steps between interval points [xStart,xFinish]
    n = 0 # initial number of steps
    # TODO: below is a bug: step size h < 1 produces evenN = False but can't reach the xFinal if the n isn't adjusted equally
    # like[0, 0.1, 0.2, 0.3 ...]
    if ((xFinish-xStart) % h == 0):
        n = ((xFinish-xStart) // h) + 2;  # print(n)
        evenN = True
    else:
        n = ((xFinish-xStart) // h) + 2;  # print(n)
        evenN = False
    n = int(n) # necessary conversion because n interpreted as a float number
    # print(n)
    yValues = np.zeros(n,dtype='float'); xMesh = np.zeros(n,dtype='float') # init mesh values
    yValues[0] = float(y0); xMesh[0] = float(xStart)
    if (evenN):
        for i in range(1,n):
            yValues[i] = yValues[i-1] + h*f(xMesh[i-1]) # simple Euler solver
            xMesh[i] = xMesh[i-1] + h
    else:
        for i in range(1,n):
            yValues[i] = yValues[i-1] + h*f(xMesh[i-1]) # simple Euler solver
            xMesh[i] = xMesh[i-1] + h
        if (xMesh[n-1] < xFinish):
            hLastStep = xFinish - (n-1)*h # Last step isn't equal to an input one
            yValues[n-1] =  yValues[n-2] + hLastStep*f(xMesh[n-2])
            xMesh[n-1] = xMesh[n-2] + hLastStep
    return (xMesh,yValues)

#%% Testing values
h = 0.2; y0 = 0.0; xStart = 0.0; xFinish = 3.0; nDigits = 2
#%% Testing
f1 = SampleF.SampleFunctions(1,nDigits); y1 = f1.getValue # For referring to a function it can be called without () notation
# print(y1.__doc__) # Check importing of docstrings as well (as part of a module documentation)
(x,y) = simpleEuler(y1,h,y0,xStart,xFinish)
# %% Theoretical values calculation
yTh = np.zeros(len(x),dtype='float')
for i in range(len(x)):
    yTh[i] = f1.getTheoreticalValue(x[i])
# %% Comparison on grahps
x.round(nDigits); y.round(nDigits)
plt.figure(); plt.xlabel("x"); plt.ylabel("y(x) solution"); plt.plot(x,y,label='numerical solution',linewidth=3)
plt.plot(x,yTh,label='theoretical function',linewidth=3); plt.legend()