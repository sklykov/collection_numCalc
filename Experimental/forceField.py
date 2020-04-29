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
        # else:
        #     print("The dipole initialized with", charge, "as both charges\n", coordinatesPlusCharge,
        #           "as coordinates of a plus charge\n", coordinatesMinusCharge, "as coordinates of a minus charge")

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
        in the thermofield (not on the central line). The dipole axis has the h approximately 1 pixel.
        Parameters
        ----------
        x, y:
            coordinates of a point of interest
        decayPower:
            power in decay power in (1/(r**(-decayPower)))
        Return
        -------
        (Ex, Ey):
            "force" values
        """
        x1 = self.coordinatesMinusCharge[0]; x2 = self.coordinatesPlusCharge[0]  # To mark equations
        y1 = self.coordinatesMinusCharge[1]; y2 = self.coordinatesPlusCharge[1]
        widthOfFlow = 1  # Important parameter that defines the the width of a flow field
        A = y1 - y2
        B = x2 - x1
        C = x1*y2 - x2*y1
        r = euclidean([x1, y1], [x2, y2])  # Length of a dipole
        decayDistance = r/2  # Characteristical decay distance = quarter of a dipole length
        # d - distance from the point to the central part of thermoflow
        d = np.absolute(A*x + B*y + C)/r  # Distance from point of interest to the main axis between "two charges"
        xPlusField = x - x2
        xMinusField = x1 - x
        yPlusField = y - y2
        yMinusField = y1 - y
        # Definition of central part of thermofield - it's orientation to the plus/minus charges
        # Actually, could be exchanged to simple code xMax = max(x1, x2); xMin = min(x1, x2)
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

        # Another attempt to correct decay of off-axis decay of aflow field
        elif(d > (widthOfFlow + 0.05)) and (x < xMax) and (x > xMin):
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
            if np.absolute(xThermoField) < self.charge:
                xThermoField *= self.equalPotentialApprox(x, y, decayPower)*np.exp(-(d)/decayDistance)
                if np.absolute(xThermoField) >= self.charge:
                    xThermoField = np.sign(xThermoField)*0.95*self.charge

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

        # d <= 1.05 due to rounding error in the distance calculations (it's especially a case in pixels calculations)
        if (d <= (widthOfFlow + 0.05)) and (y < yMax) and (y > yMin):
            yThermoField = self.charge*np.sign(yPlusField + yMinusField)

        # Another attempt to correct decay of off-axis decay of aflow field
        elif(d > (widthOfFlow + 0.05)) and (y < yMax) and (y > yMin):
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
            if np.absolute(yThermoField) < self.charge:
                yThermoField *= self.equalPotentialApprox(x, y, decayPower)*np.exp(-(d)/decayDistance)
                if np.absolute(yThermoField) >= self.charge:
                    yThermoField = np.sign(yThermoField)*0.95*self.charge

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

        # Rounding only for more clear observing equal force lines!
        xThermoField = round(xThermoField, 1)
        yThermoField = round(yThermoField, 1)
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

    def equalPotentialApprox(self, x: float, y: float, decayPower: float):
        """
        Approximation of calculation of force lines with equal forces along them.
        Parameters
        ----------
        x, y:
            coordinates of point of interest, there the field should be calculated
        """
        thermoFieldAdjustment = 1
        x1 = self.coordinatesMinusCharge[0]
        x2 = self.coordinatesPlusCharge[0]
        y1 = self.coordinatesMinusCharge[1]
        y2 = self.coordinatesPlusCharge[1]
        xMin = min(x1, x2)
        yMin = min(y1, y2)
        xDipoleCenter = (np.absolute(x1 - x2)/2) + xMin
        yDipoleCenter = (np.absolute(y1 - y2)/2) + yMin
        distanceCenterPoint = euclidean([xDipoleCenter, yDipoleCenter], [x, y])
        # print("dipole center at", [xDipoleCenter, yDipoleCenter])
        Adipole = float(y1 - y2)
        Bdipole = float(x2 - x1)
        # Dipole length l:
        L = euclidean([x1, y1], [x2, y2])
        # print(L, "- dipole length")
        # All cryptic names like A, B, k belongs to equations of a line in 2D space: y = k*x + b; A*x + B*x + C = 0
        maxMultiplicator = 2*np.power(L, decayPower) / (self.charge*np.power(2, decayPower))
        Aline = float(yDipoleCenter - y)
        Bline = float(x - xDipoleCenter)
        if (Bline != 0):  # It means that line connecting center of a dipole and a point is parallel to X axis
            kLine = -(Aline/Bline)
        else:
            kLine = 0
        if (Bdipole != 0):  # It means that dipole is parallel to X axis
            kDipole = -(Adipole/Bdipole)
        else:
            kDipole = 0
        # So, actually kLine == 0 when either it's parallel or perpendicular to the X axis
        # print(Adipole, Bdipole, kDipole, " - A, B, k of a dipole")
        # print(Aline, Bline, kLine, " - A, B, k of a line")
        if (kDipole == 0):
            # Dipole is parallel to some of axis - X or Y
            if (kLine != 0):
                # The line has some inclination to a dipole
                # print("Dipole is parallel to some axis. A line - not")
                gamma = np.arctan(kLine)
                thermoFieldAdjustment = 1 + np.absolute(np.sin(gamma))*(maxMultiplicator - 1)  # Perpendiculars enhance
                thermoFieldAdjustment += np.absolute(np.cos(gamma))*((L/4)/distanceCenterPoint)*(maxMultiplicator - 1)
            else:
                # Both dipole and line are parallel to some axis
                if (Aline == 0 and Bdipole == 0) or (Bline == 0 and Adipole == 0):
                    # they should be perpendicular to each other
                    # print("Dipole is parallel to some axis. A line - perpendicular to it")
                    thermoFieldAdjustment = maxMultiplicator
                else:
                    # they should be parallel
                    # print("Dipole is parallel to some axis. A line - parallel to it")
                    thermoFieldAdjustment = 1
        else:
            # Two lines - dipole and for coordinate are perpendicular to each other
            if (kLine == -(1/kDipole)):
                # print("Dipole and a line are perpendicular to each other")
                thermoFieldAdjustment = maxMultiplicator
            elif (kDipole == kLine):
                # Two lines - dipole and for coordinate are parallel to each other
                # print("Dipole and a line are parallel to each other")
                thermoFieldAdjustment = 1
            else:
                # print("Dipole and a line have some inclination between each other")
                gamma = np.arctan((kLine - kDipole)/(1 + kLine*kDipole))
                thermoFieldAdjustment = 1 + np.absolute(np.sin(gamma))*(maxMultiplicator - 1)
                # For mostly parallel points to dipole axis but that aren't enhanced
                thermoFieldAdjustment += np.absolute(np.cos(gamma))*((L/4)/distanceCenterPoint)*(maxMultiplicator - 1)

        return thermoFieldAdjustment
