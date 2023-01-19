# -*- coding: utf-8 -*-
"""
Euler - Cromer algorithm for solving 2nd order ODEs like y''(x) = f(x,t,y(x)).

And 1D oscillator modelling as the demo.

@author: sklykov
@license: The Unlicense
"""
# %% Import section
import numpy as np
import matplotlib.pyplot as plt
from DynamicSystem import dynamicSys


# %% Algorithm implementation (in general form - using 1st, 2nd order derivatives)
def eulerCromer(f: float, y1stDerivPrevious: float, yPrevious: float, h: float) -> tuple:
    """Implement is kept general - it consumes only the values calculated outside of it."""
    y2ndDeriv = f  # y''(x..) = f(*args); a  = f/m - this is the 2nd derivative!!! (acceleration)
    y1stDeriv = y1stDerivPrevious + h*y2ndDeriv  # y'(n) = y'(n-1) + h*y''; this is speed or velocity!
    y = yPrevious + h*y1stDeriv
    return (y1stDeriv, y)

# %% Testing values and their probing
gamma = 0.015; radius = 5; mass = 1; amplitude = 8.8
xStart = -2*radius; xFinish = 2*radius
initSpeed = 1.6
nTimePoints = int(40000); hTime = 0.0025
sys1 = dynamicSys("1st example"); sysForce1 = sys1.getForce

# %% Getting force profile
nX = 1000; hX = (xFinish - xStart)/nX
nX += 1
f = np.zeros(nX, dtype=float); x = np.zeros(nX, dtype=float)
for i in range(nX):
    x[i] = xStart + i*hX
    gammaPureExternalForce = 0.0
    f[i] = sysForce1(amplitude, x[i], radius, mass, gammaPureExternalForce, initSpeed)

plt.figure(); plt.plot(x, f, linewidth=2); plt.xlabel("x coordinates"); plt.title("External force profile f(x)")

# %% Testing algorithm for solving 1D oscillatory form
xCoord = np.zeros(nTimePoints+1, dtype='float'); speeds = np.zeros(nTimePoints+1, dtype='float')
timePoints = np.zeros(nTimePoints+1, dtype='float')
xCoord[0] = xStart; speeds[0] = initSpeed; timePoints[0] = 0
# Get modelled values - x coordinates, speed along x
for i in range(1, nTimePoints+1):
    forceCalibrated = sysForce1(amplitude, xCoord[i-1], radius, mass, gamma, speeds[i-1])
    (speeds[i], xCoord[i]) = eulerCromer(forceCalibrated, speeds[i-1], xCoord[i-1], hTime)
    timePoints[i] = timePoints[i-1] + hTime
# Plotting dynamic pictures for x and v
plt.figure(); plt.plot(timePoints, xCoord, 'r', linewidth=2); plt.xlabel("time"); plt.ylabel("x"); plt.title("x(t)")
plt.figure(); plt.plot(timePoints, speeds, 'g', linewidth=2); plt.xlabel("time"); plt.ylabel("v"); plt.title("v(t)")
plt.figure(); plt.plot(xCoord, speeds, 'm', linewidth=1); plt.xlabel("x"); plt.ylabel("v")
plt.title("speed versus x coordinate")
