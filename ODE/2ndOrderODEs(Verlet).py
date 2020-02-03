# -*- coding: utf-8 -*-
"""
The velocity Verlet method for 2nd order ODE solving like y''(x) = f(x,t,y(x))
The demo is copied from the Euler-Cromer method implementation (1D oscillator)
@author: ssklykov
"""
# %% Import section
import numpy as np
import matplotlib.pyplot as plt
from DynamicSystem import dynamicSys

# %% The first part of the algorithm implementation - calculation only y - returning for another function
def verletYCalc(y2ndDerivPrevious:float,y1stDerivPrevious:float,yPrevious:float,h:float) -> tuple:
    """1st part of algorithm - demands only precalculated values, returning y(m+1) and y'(m+1/2)"""
    y1stDerivHalf = y1stDerivPrevious + 0.5*h*y2ndDerivPrevious # y'(m+1/2) = ...
    y = yPrevious + h*y1stDerivHalf
    return (y,y1stDerivHalf)

# %% Other part of the algorithm implementation - calculation of y' and y''
def verletDerivCalc(function:float,y1stDerivHalf:float,h:float) -> tuple:
    """2nd part of algorithm - demands f(x+h,y(m+1),y'(m+1/2))"""
    y2ndDeriv = function # simple assignment
    y1stDeriv = y1stDerivHalf + 0.5*h*y2ndDeriv
    return (y1stDeriv,y2ndDeriv)


# %% Testing values and their probing
gamma = 0.025; radius = 5; mass = 1; amplitude = 10.8
xStart = -2*radius; xFinish = 2*radius
initSpeed = 2.2
nTimePoints = int(45000); hTime = 0.004
sys1 = dynamicSys("1st example"); sysForce1 = sys1.getForce

# %% Getting force profile - in the Euler - Cromer demo

# %% Testing algorithm for solving 1D oscillatory form
xCoord = np.zeros(nTimePoints+1,dtype='float'); speeds = np.zeros(nTimePoints+1,dtype='float')
timePoints = np.zeros(nTimePoints+1,dtype='float'); accelerations = np.zeros(nTimePoints+1,dtype='float')
# Initial values at 0th time point (t0 = 0)
xCoord[0] = xStart; speeds[0] = initSpeed; timePoints[0] = 0
accelerations[0] = sysForce1(amplitude,xCoord[0],radius,mass,gamma,speeds[0])
# Get modelled values - x coordinates, speed along x
for i in range(1,nTimePoints+1):
    forceCalibrated = sysForce1(amplitude,xCoord[i-1],radius,mass,gamma,speeds[i-1])
    (xCoord[i],speedInterim) = verletYCalc(accelerations[i-1],speeds[i-1],xCoord[i-1],hTime)
    accelCurrent = sysForce1(amplitude,xCoord[i],radius,mass,gamma,speedInterim)
    (speeds[i],accelerations[i]) = verletDerivCalc(accelCurrent,speedInterim,hTime)
    timePoints[i] = timePoints[i-1] + hTime

# Plotting dynamic pictures for x and v
plt.figure(); plt.plot(timePoints,xCoord,'r',linewidth=2); plt.xlabel("time"); plt.ylabel("x"); plt.title("x(t)")
plt.figure(); plt.plot(timePoints,speeds,'g',linewidth=2); plt.xlabel("time"); plt.ylabel("v"); plt.title("v(t)")
plt.figure(); plt.plot(xCoord,speeds,'m',linewidth=1); plt.xlabel("x"); plt.ylabel("v"); plt.title("speed versus x coordinate")
