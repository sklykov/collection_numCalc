# -*- coding: utf-8 -*-
"""
Testing of forceField class capabilities.
The 'main' file for testing forceField.py, particlesDynamics.py
@author: ssklykov
"""
# %% Imports
from forceField import dipole
from particlesDynamics import pDynamics

# %% Parameters
charge = 150
xPlusCharge = 4
yPlusCharge = 4
xMinusCharge = 36
yMinusCharge = 36
size = 40
nPoints = size
decayPower = 2
nParticles = 2
diffusionPower = 1
# If particle is located on the main dipole / thermofield axis, then each maximum step equal to it
tStep = float(charge / 800)
precision_digits = 0

# %% Demo
dip = dipole(charge, [xPlusCharge, yPlusCharge], [xMinusCharge, yMinusCharge])
# (Ex, Ey) = dip1.createFieldMap(size, nPoints)
(ExT, EyT) = dip.createFieldMap(size, nPoints, "thermo", decayPower)
particles = pDynamics(size, nParticles)
x = particles.coordinatesX
y = particles.coordinatesY
print(dip.equalPotentialApprox(14, 14), " - correction coefficient")
print(dip.equalPotentialApprox(15, 17), " - correction coefficient")

# for i in range(2):
#     for j in range(nParticles):
#         coordinates = [x[j], y[j]]
#         print(coordinates, "before force field application")
#         (x[j], y[j]) = pDynamics.dynamicsInertionless(
#             coordinates, dip.thermoFieldCalc(x[j], y[j], decayPower), diffusionPower, tStep, precision_digits)
#         print(x[j], y[j], "after applied force field")
