# -*- coding: utf-8 -*-
"""
Testing of forceField class capabilities.

@author: ssklykov
"""
# %% Imports
from forceField import dipole

# %% Parameters
charge = 100
xPlusCharge = 4
yPlusCharge = 4
xMinusCharge = 36
yMinusCharge = 36
size = 40
nPoints = 40

# %% Demo
dip1 = dipole(charge, [xPlusCharge, yPlusCharge], [xMinusCharge, yMinusCharge])
# (Ex, Ey) = dip1.createFieldMap(size, nPoints)
(Ex, Ey) = dip1.createFieldMap(size, nPoints)
(ExT, EyT) = dip1.createFieldMap(size, nPoints, "thermo")
