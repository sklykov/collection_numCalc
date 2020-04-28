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
        distanceMinus = euclidean(self.coordinatesMinusCharge, [x, y])  # Distance between - charge and point in 2D dim
        # plusFieldModule = self.charge*np.exp(-(distancePlus)/4)  # Modulus of field tension (hypothetic)
        # minusFieldModule = self.charge*np.exp(-(distanceMinus)/4)  # Modulus of field tension
        plusFieldModule = self.charge  # Modulus of field tension (mimicking r^(-2) decay)
        if distancePlus > 1:
            plusFieldModule *= 1/(np.power(distancePlus, 2))
        minusFieldModule = self.charge  # Modulus of field tension (mimicking r^(-2) decay)
        if distanceMinus > 1:
            minusFieldModule *= 1/(np.power(distanceMinus, 2))
        xPlusField = plusFieldModule*(x - self.coordinatesPlusCharge[0])  # Hypothetic repulsive field (plus charge)
        yPlusField = plusFieldModule*(y - self.coordinatesPlusCharge[1])
        xMinusField = minusFieldModule*(self.coordinatesMinusCharge[0] - x)  # Hypothetic attractive field (minus)
        yMinusField = minusFieldModule*(self.coordinatesMinusCharge[1] - y)
        xDipoleField = xPlusField + xMinusField  # Sum of plus and minus components of force fields
        yDipoleField = yPlusField + yMinusField
        return (xDipoleField, yDipoleField)

    def thermoFieldCalc(self, x: float, y: float, decayPower: float) -> tuple:
        """Mimicking the force field (thermo flow) created by scanning laser in liquid medium.
        Charge gives the estimate of force tension. The parameter decayPower helps to calculate decay power (^ - power)
        in the thermofield (not on the central line). The dipole axis has the h approximately 1 pixel."""
        x1 = self.coordinatesMinusCharge[0]; x2 = self.coordinatesPlusCharge[0]  # To mark equations
        y1 = self.coordinatesMinusCharge[1]; y2 = self.coordinatesPlusCharge[1]
        widthOfFlow = 1  # Important parameter that defines the the width of a flow field
        # decayDistance = 10
        A = y1 - y2
        B = x2 - x1
        C = x1*y2 - x2*y1
        r = euclidean([x1, y1], [x2, y2])
        # d - distance from the point to the central part of thermoflow
        d = np.absolute(A*x + B*y + C)/r  # Distance from point of interest to the main axis between "two charges"
        xPlusField = x - x2
        xMinusField = x1 - x
        yPlusField = y - y2
        yMinusField = y1 - y
        # Definition of central part of thermofield - it's orientation to the plus/minus charges
        if (x2 < x1):
            xMax = x1
            xMin = x2
        else:
            xMax = x2
            xMin = x1
        if (y2 < y1):
            yMax = y1
            yMin = y2
        else:
            yMax = y2
            yMin = y1
        # For making always right direction - the orientation depends on the orientations of minus / plus charges
        # This is central part of thermoflow
        if (d <= (widthOfFlow + 0.05)) and (x <= xMax) and (x >= xMin):
            xThermoField = self.charge*np.sign(xPlusField + xMinusField)

        # Below - the attempt to reduce decay parralel to the flow field axis by multiplying to the emperic function
        # elif (d > (widthOfFlow + 0.05)) and (x < xMax) and (x > xMin):
        #     distancePlus = euclidean(self.coordinatesPlusCharge, [x, y])  # Distance between + charge and point in 2D
        #     plusFieldModule = self.charge
        #     thermoFieldAdjustment = 1  # To correct strong decay out of flow axis
        #     if distancePlus > 1:
        #         plusFieldModule *= 1/(np.power(distancePlus, decayPower))
        #     minusFieldModule = self.charge
        #     distanceMinus = euclidean(self.coordinatesMinusCharge, [x, y])
        #     if distanceMinus > 1:
        #         minusFieldModule *= 1/(np.power(distanceMinus, decayPower))
        #     if (distanceMinus > 4) and (distancePlus > 4):
        #         thermoFieldAdjustment = 1/(np.sqrt(abs(d-widthOfFlow)))
        #     xPlusField *= plusFieldModule
        #     xMinusField *= minusFieldModule
        #     xThermoField = -(xPlusField + xMinusField)
        #     xThermoField *= thermoFieldAdjustment

        else:
            distancePlus = euclidean(self.coordinatesPlusCharge, [x, y])  # Distance between + charge and point in 2D
            plusFieldModule = self.charge
            if distancePlus > 1:
                plusFieldModule *= 1/(np.power(distancePlus, decayPower))
            minusFieldModule = self.charge
            distanceMinus = euclidean(self.coordinatesMinusCharge, [x, y])  # Distance between + charge and point in 2D
            if distanceMinus > 1:
                minusFieldModule *= 1/(np.power(distanceMinus, decayPower))
            xPlusField *= plusFieldModule
            xMinusField *= minusFieldModule
            xThermoField = -(xPlusField + xMinusField)
            xThermoField = round(xThermoField, 1)
            # if np.absolute(xThermoField) < self.charge:
            #     xThermoField *= 3*np.exp(-(d)/decayDistance)
            #     if np.absolute(xThermoField) >= self.charge:
            #         xThermoField = np.sign(xThermoField)*0.95*self.charge

        # d <= 1.05 due to rounding error in the distance calculations (it's especially a case in pixels calculations)
        if (d <= (widthOfFlow + 0.05)) and (y <= yMax) and (y >= yMin):
            yThermoField = self.charge*np.sign(yPlusField + yMinusField)

        # Attempt to reduce decay
        # elif (d > (widthOfFlow + 0.05)) and (y < yMax) and (y > yMin):
        #     distancePlus = euclidean(self.coordinatesPlusCharge, [x, y])  # Distance between + charge and point in 2D
        #     plusFieldModule = self.charge
        #     thermoFieldAdjustment = 1  # To correct strong decay out of flow axis
        #     if distancePlus > 1:
        #         plusFieldModule *= 1/(np.power(distancePlus, decayPower))
        #     minusFieldModule = self.charge
        #     distanceMinus = euclidean(self.coordinatesMinusCharge, [x, y])
        #     if distanceMinus > 1:
        #         minusFieldModule *= 1/(np.power(distanceMinus, decayPower))
        #     if (distanceMinus > 4) and (distancePlus > 4):
        #         thermoFieldAdjustment = 1/(np.sqrt(abs(d-widthOfFlow)))
        #     yPlusField *= plusFieldModule
        #     yMinusField *= minusFieldModule
        #     yThermoField = -(yPlusField + yMinusField)
        #     yThermoField *= thermoFieldAdjustment

        # Here the field should resemble the dipole far field
        else:
            distancePlus = euclidean(self.coordinatesPlusCharge, [x, y])  # Distance between + charge and point in 2D
            plusFieldModule = self.charge
            if distancePlus > 1:
                plusFieldModule *= 1/(np.power(distancePlus, decayPower))
            minusFieldModule = self.charge
            distanceMinus = euclidean(self.coordinatesMinusCharge, [x, y])  # Distance between + charge and point in 2D
            if distanceMinus > 1:
                minusFieldModule *= 1/(np.power(distanceMinus, decayPower))
            yPlusField *= plusFieldModule
            yMinusField *= minusFieldModule
            yThermoField = -(yPlusField + yMinusField)
            yThermoField = round(yThermoField, 1)
            # if np.absolute(yThermoField) < self.charge:
            #     yThermoField *= 3*np.exp(-(d)/decayDistance)
            #     if np.absolute(yThermoField) >= self.charge:
            #         yThermoField = np.sign(yThermoField)*0.95*self.charge

        return (xThermoField, yThermoField)

    def createFieldMap(self, size: int, nPoints: int, fieldType: str = "dipole", decayPower: float = 1.5):
        """Shows the map of created by a dipole the force field (can be calculated by using 'forceFieldCalc' method
        The map itself is a square with sizes [dimension x dimension]. Two types of force fields: 'dipole'
        (default) and 'thermo'.
        Parameters
        ----------
        size:
            size of generated map of a force field
        nPoints:
            number of points for calculation of a force (tension) in them
        decayPower:
            demanded by 'thermo' field parameter
        Returns
        -------
        tuple with meschgrids of Ex, Ey - tensions (forces) of a selected force field.
        """
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
                if fieldType == "dipole":
                    (Ex[i][j], Ey[i][j]) = self.forceFieldCalc(xP, yP)
                elif fieldType == "thermo":
                    (Ex[i][j], Ey[i][j]) = self.thermoFieldCalc(xP, yP, decayPower)
        fig1 = plt.figure(figsize=(8, 8))  # Empty figure
        fig1.tight_layout()
        plt.streamplot(x, y, Ex, Ey, linewidth=1)  # Creation of map of force field (streamlines of a vector flow)
        fig2 = plt.figure(figsize=(8, 8))
        fig2.tight_layout()
        plt.quiver(x, y, Ex, Ey)  # Over type of force field mapping
        return (Ex, Ey)

    def simpleScalingFactor(self, maxDisplacement: float) -> float:
        """
        Attempt to provide some scaling factor (time step) for making a simulations with pixel values in the case of
        non-physical simulations (not real diffusion coefficients, )
        Parameters
        ----------
        maxDisplacement : float
            maximum desired displacements in pixels for simulations

        Returns
        -------
        scaling_factor(or time step):
            the value that could be provided for a simple dynamic equation: dx = force[x,y]*scaling_factor

        """
        scaling_factor = maxDisplacement / self.charge
        return scaling_factor

    def equalPotentialApprox(self, x: float, y: float):
        """
        Approximation of calculation of force lines with equal forces along them.
        """
        thermoFieldAdjustment = 1
        x1 = self.coordinatesMinusCharge[0]
        x2 = self.coordinatesPlusCharge[0]
        y1 = self.coordinatesMinusCharge[1]
        y2 = self.coordinatesPlusCharge[1]
        xDipoleCenter = np.absolute(x1 - x2)/2
        yDipoleCenter = np.absolute(y1 - y2)/2
        # print("dipole center at", [xDipoleCenter, yDipoleCenter])
        Adipole = y1 - y2
        Bdipole = x2 - x1
        # Dipole length l:
        L = euclidean([x1, y1], [x2, y2])
        maxMultiplicator = (L*L)/(4*self.charge)
        Aline = yDipoleCenter - y
        Bline = x - xDipoleCenter
        if (Bline != 0):
            kLine = -(Aline/Bline)
        else:
            kLine = 0
        if (Bdipole != 0):
            kDipole = -(Adipole/Bdipole)
        else:
            kDipole = 0
        print(Adipole, Bdipole, kDipole, " - A, B, k of a dipole")
        print(Aline, Bline, kLine, " - A, B, k of a line")
        if (kLine == kDipole):
            # Two lines - dipole and for coordinate are parallel to each other
            thermoFieldAdjustment = 1
        elif (kLine == -(1/kDipole)) or (kDipole == 0 and kLine == 0):
            # Two lines - dipole and for coordinate are perpendicular to each other
            thermoFieldAdjustment = maxMultiplicator

        return thermoFieldAdjustment
