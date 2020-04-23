# -*- coding: utf-8 -*-
"""
Testing of forceField class capabilities.

@author: ssklykov
"""
# %% Imports
from forceField import dipole

# %% Parameters
charge = 100
xPlusCharge = 20
yPlusCharge = 20
xMinusCharge = 60
yMinusCharge = 60
size = 100
nPoints = 100

# %% Demo
dip1 = dipole(charge, [xPlusCharge, yPlusCharge], [xMinusCharge, yMinusCharge])
# (Ex, Ey) = dip1.createFieldMap(size, nPoints)
dip1.createFieldMap(size, nPoints)