# -*- coding: utf-8 -*-
"""
Calculation of dynamics of a hypothetic particle guided by a generated force field.

Implementation - in a dedicated class below.

@author: sklykov
@license: The Unlicense
"""
# %% Imports
import numpy as np


# %% Class implementation for simulation of particle dynamics
class pDynamics():
    """The main of this class - to simulate dynamics of particles floating in some viscous liquid medium and
    influenced by some generated force field (see example of implementation in 'forceField.py'.
    """
    # Default values for showing them explicitly
    particles_number = 1
    sizeOfPicture = 100
    coordinatesX = np.ndarray(particles_number, dtype='float')
    coordinatesY = np.ndarray(particles_number, dtype='float')

    def __init__(self, sizeOfPicture: float, particles_number: int = 1):
        """The main aim - to initialize class with positive number of particles and size of a picture - there this
        particles will be spread out.
        Parameters:
            sizeOfPicture - in pixels or in physical values (must be positive!)
            particles_number - number of particles
        """
        if (particles_number < 0):
            print("Provided negative number of particles. It's reversed")
            particles_number *= -1
        elif (particles_number == 0):
            print("Provided zero particles. It's assigned to default value = 1")
            particles_number = 1
        self.particles_number = particles_number
        if (sizeOfPicture < 0):
            print("Provided negative size of a pictures. It's reversed")
            sizeOfPicture *= -1
        elif (sizeOfPicture == 0):
            print("Provided zero sizeOfPicture. It's assigned to default value = 100")
            sizeOfPicture = 100
        self.sizeOfPicture = sizeOfPicture
        coordinatesX = np.ndarray(particles_number, dtype='float')
        coordinatesY = np.ndarray(particles_number, dtype='float')
        for i in range(particles_number):
            coordinatesX[i] = np.random.randint(0, sizeOfPicture)
            coordinatesY[i] = np.random.randint(0, sizeOfPicture)
        self.coordinatesX = coordinatesX
        self.coordinatesY = coordinatesY

    @staticmethod
    def diffusionStep(diffusionPower: float = 2, rounding_number: int = 1) -> float:
        """Simulation of random walk (diffusion).
        Parameters
        ----------
        diffusionPower:
            for calculation of magnitude of a random step (+ dx)
        rounding_number:
            integer for round() function as returning coordinate (necessary, if coordinates are integers in the end)
        Return
        ------
        diffision step
            for further summing with x or y coordinate
        """
        diffusionStep = (np.random.random() - 0.5)*2  # Generation of random number from the interval [-1; 1)
        diffusionStep *= diffusionPower  # Generating of diffusion step accounting the Diffusion coefficient
        return round(diffusionStep, rounding_number)

    @staticmethod
    def dynamicsInertionless(coordinates: list, force: list, diffusionPower: float,
                             time_step: float, precision_digits: int) -> tuple:
        """Simple integration of intertionless dynamics of particle. Viscous properties of medium isn't accounted.
        Just a simple model for my purposes. Maybe, will be developed to more physically relevant module in future.
        Parameters
        ----------
        coordinates:
            [x, y]
        force:
            force at initial point [x, y], pushing the particle
        time_step:
            for scaling of force field applying
        """
        x = coordinates[0]
        y = coordinates[1]
        # Force pushing particle directly, without approximation of time interval (regulated by the calling function)
        x += force[0]*time_step
        x += pDynamics.diffusionStep(diffusionPower, 2)
        y += force[1]*time_step
        y += pDynamics.diffusionStep(diffusionPower, 2)
        x = round(x, precision_digits)
        y = round(y, precision_digits)
        return (x, y)

    @staticmethod
    def unpackCoordinates(coordinates: list) -> tuple:
        """Unpacking coordinates from a list [x1, y1, ...] to two separate arrays."""
        nParticles = len(coordinates) // 2
        x = np.ndarray(nParticles, dtype='float')
        y = np.ndarray(nParticles, dtype='float')
        for i in range(nParticles):
            x[i] = coordinates[2*i]
            y[i] = coordinates[2*i + 1]
        return (x, y)

    @staticmethod
    def packCoordinates(x, y) -> list:
        """Unpacking coordinates from arrays like [x1, x2, ...], [y1, y2, ...] to an one merged list."""
        n = 2*len(x)
        composedList = [0]*n
        for i in range(len(x)):
            composedList[2*i] = x[i]
            composedList[2*i + 1] = y[i]
        return composedList
