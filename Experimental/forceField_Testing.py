# -*- coding: utf-8 -*-
"""
Testof forceField class capabilities.

The 'main' file for testing forceField.py, particlesDynamics.py

@author: sklykov
@license: The Unlicense
"""
# %% Imports
from forceField import dipole
from particlesDynamics import pDynamics

# %% Parameters
charge = 150
xPlusCharge = 36
yPlusCharge = 4
xMinusCharge = 4
yMinusCharge = 36
size = 50
nPoints = size
decayPower = 1.5
nParticles = 2
diffusionPower = 2
widthOfFlow = 2
# If particle is located on the main dipole / thermofield axis, then each maximum step equal to it
tStep = float(charge / 800)
precision_digits = 1
maxDisplacementInPixels = 10

# %% Demo
dip = dipole(charge, [xPlusCharge, yPlusCharge], [xMinusCharge, yMinusCharge])
# (Ex, Ey) = dip1.createFieldMap(size, nPoints)
(ExT, EyT) = dip.createFieldMap(size, nPoints, "dipole", decayPower, widthOfFlow)
particles = pDynamics(size, nParticles)
x = particles.coordinatesX
y = particles.coordinatesY
# print(dip.equalPotentialApprox(18, 22, decayPower), " - correction coefficient")
# print(dip.equalPotentialApprox(16, 16), " - correction coefficient")
# print(dip.equalPotentialApprox(19, 23, decayPower), " - correction coefficient")
scaling_factor = round(dip.simpleScalingFactor(maxDisplacementInPixels), 3)
print(scaling_factor, "- possible size of time step for simple dynamic equation")

for i in range(3):
    print((x, y), "before force field application")
    dip.updateCoordinates(decayPower, x, y, diffusionPower, scaling_factor, precision_digits, widthOfFlow)
    print((x, y), "after force field application")
    # for j in range(nParticles):
    #     coordinates = [x[j], y[j]]
    #     print(coordinates, "before force field application")
    #     (x[j], y[j]) = pDynamics.dynamicsInertionless(
    #         coordinates, dip.thermoFieldCalc(x[j], y[j], decayPower),
    #         diffusionPower, scaling_factor, precision_digits)
    #     print(x[j], y[j], "after applied force field")
overallCoordinates = pDynamics.packCoordinates(x, y)
