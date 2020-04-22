# -*- coding: utf-8 -*-
"""
Experiments with creation of attractors / repulsers - something resembling the electrostatic field.
@author: ssklykov
"""
# %% Imports
import numpy as np


# %% Class definition
class dipole():
    """Simulation of electrostatic dipole (only mimicking). """
    coordinatesPlusCharge = np.array([0, 0])
    coordinatesMinusCharge = np.array([1, 1])

    def __init__(self, coordinatesPlusCharge, coordinatesMinusCharge):
        """Coordinates transferred as in dimensionless units! No mm, um... Pixel like. 2D case."""
        self.coordinatesPlusCharge = coordinatesPlusCharge
        self.coordinatesMinusCharge = coordinatesMinusCharge
        if (len(coordinatesPlusCharge) != 2) or (len(coordinatesMinusCharge) != 2):
            raise("Not 2D Coordinates provided")

    def forceFieldCalc():
        pass
