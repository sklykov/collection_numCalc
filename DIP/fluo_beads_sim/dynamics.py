# -*- coding: utf-8 -*-
"""
Simulation of diffusion / motion of microbeads.

@author: ssklykov
"""
# %% Imports
import numpy as np


# %% Class definition
class diffusion():
    D = 1.0  # Diffusion coefficient
    coordinates = []
    time_interval = 1.0  # could be in different physical values
    calibration = 1.0

    def __init__(self, initial_point: list, D: float = 1.0, time_interval: float = 1.0, calibration: float = 1.0):
        """
        Initialization of the class for generation of 2D diffusion movement.

        Parameters
        ----------
        initial_point : list
            Initial coordinates of a bead in the form [x, y] or [i, j].
        D : float, optional
            Diffusion coefficient. Take care for selection of its physical units! The default is 1.0.
        time_interval : float, optional
            Time interval between steps or simulated frames. The default is 1.0.
        calibration: float, optional
            Calibration parameter for transferring steps in nm or um into pixels for drawing a bead on a scene.
            The default is 1.0.

        Raises
        ------
        ValueError
            If initial point is specified in the form other than [x: Number, y: Number].

        Returns
        -------
        None.

        """
        self.D = D
        self.time_interval = time_interval
        self.calibration = calibration
        if len(initial_point) == 2:
            self.coordinates.append(initial_point)
        else:
            raise ValueError("Initial point should contain initial coordinates of a bead in form like [x, y]")

    def get_next_point(self) -> list:
        # Simulation for each coordinate in the approximated relation: r ~= sqrt(2)*x or r ~= sqrt(2)*y
        # The equation is taken from Qian, et al. (1991) in simplified form, when origin is on the previous position
        sigma = np.sqrt(2.0*self.D*self.time_interval)
        k = 1/(2*sigma*np.sqrt(np.pi))  # coefficient for conformity with the equation from Qian, et al. (1991)
        # print(sigma, k)
        x_step = k*np.random.normal(0, sigma)
        x_step *= self.calibration
        y_step = k*np.random.normal(0, sigma)
        y_step *= self.calibration
        # rounding for trimming unnecessary precision
        x_step = np.round(x_step, 3)
        y_step = np.round(y_step, 3)
        # print(x_step, y_step)
        x_next = self.coordinates[len(self.coordinates)-1][0] + x_step
        y_next = self.coordinates[len(self.coordinates)-1][1] + y_step
        # print(x_next, y_next)
        self.coordinates.append([x_next, y_next])
        return [x_next, y_next]


# %% Testing
if __name__ == '__main__':
    diff = diffusion([0, 0])
    diff.get_next_point()
