# -*- coding: utf-8 -*-
"""
Experiments with creation of attractors / repulsers - something resembling the electrostatic field.
@author: ssklykov
"""
# %% Imports
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# %% Class definition
class dipole():
    """Simulation of electrostatic dipole (only mimicking). """
    coordinatesPlusCharge = np.array([0, 0])
    coordinatesMinusCharge = np.array([1, 1])
    charge = 10

    def __init__(self, charge: float, coordinatesPlusCharge, coordinatesMinusCharge):
        """Coordinates transferred as in dimensionless units! No mm, um... Pixel like. 2D case.
        Charges - equal for plus and minus """
        self.coordinatesPlusCharge = coordinatesPlusCharge
        self.coordinatesMinusCharge = coordinatesMinusCharge
        self.charge = charge
        if (len(coordinatesPlusCharge) != 2) or (len(coordinatesMinusCharge) != 2):
            raise("Not 2D Coordinates provided")
        else:
            print("The dipole initialized with", charge, "as both charges\n", coordinatesPlusCharge,
                  "as coordinates of a plus charge\n", coordinatesMinusCharge, "as coordinates of a minus charge")

    def forceFieldCalc(self, x: float, y: float) -> tuple:
        """Calculation of hypothetic force field between two equal charges - plus and minus.
        Charge - hypothetic, more suitable as eletrostatic constant * charge, non-physical."""
        distancePlus = euclidean(self.coordinatesPlusCharge, [x, y])  # Distance between + charge and point in 2D dim
        plusFieldModule = self.charge*np.exp(-(distancePlus)/4)  # Modulus of field tension (hypothetic)
        xPlusField = plusFieldModule*(x - self.coordinatesPlusCharge[0])  # Hypothetic repulsive field (plus charge)
        yPlusField = plusFieldModule*(y - self.coordinatesPlusCharge[1])
        distanceMinus = euclidean(self.coordinatesMinusCharge, [x, y])  # Distance between - charge and point in 2D dim
        minusFieldModule = self.charge*np.exp(-(distanceMinus)/4)  # Modulus of field tension
        xMinusField = minusFieldModule*(self.coordinatesMinusCharge[0] - x)  # Hypothetic attractive field (minus)
        yMinusField = minusFieldModule*(self.coordinatesMinusCharge[1] - y)
        xDipoleField = xPlusField + xMinusField  # Sum of plus and minus components of force fields
        yDipoleField = yPlusField + yMinusField
        return (xDipoleField, yDipoleField)

    def createFieldMap(self, size: int, nPoints: int):
        """Shows the map of created by a dipole the force field (can be calculated by using 'forceFieldCalc' method
        The map itself is a square with sizes [dimension x dimension]."""
        # step = size/nPoints  # Step for making grid (dimensionless, "pixels")
        xGrid = np.linspace(0, size, nPoints)
        # print(xGrid)
        yGrid = np.linspace(0, size, nPoints)
        x, y = np.meshgrid(xGrid, yGrid)
        # print(x)
        ExGrid, EyGrid = np.zeros(nPoints, dtype='float'), np.zeros(nPoints, dtype='float')
        Ex, Ey = np.meshgrid(ExGrid, EyGrid)
        for i in range(nPoints):
            for j in range(nPoints):
                xP = x[i][j]
                yP = y[i][j]
                (Ex[i][j], Ey[i][j]) = self.forceFieldCalc(xP, yP)
        fig1 = plt.figure()  # Empty figure
        plt.streamplot(x, y, Ex, Ey, linewidth=1)  # Creation of map of force field (streamlines of a vector flow)
        fig2 = plt.figure()
        plt.quiver(x, y, Ex, Ey)
        # return (Ex, Ey)
