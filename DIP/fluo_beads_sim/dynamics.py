# -*- coding: utf-8 -*-
"""
Simulation of diffusion / motion of microbeads.

@author: ssklykov
"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import os

# %% Class definition
class diffusion():
    D = 1.0  # Diffusion coefficient
    coordinates = []
    time_interval = 1.0  # could be in different physical values
    calibration = 1.0
    x_steps = np.zeros(1000, dtype=float)
    y_steps = np.zeros(1000, dtype=float)
    r = np.zeros(1000, dtype=float)
    x_generated_steps = []
    y_generated_steps = []
    r_generated = []

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
            So, the unit is [pixel/nm or pixel/um]. The default is 1.0.

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
            # print(self.coordinates)
        else:
            raise ValueError("Initial point should contain initial coordinates of a bead in form like [x, y]")

    def get_next_point(self) -> list:
        # Simulation for each coordinate in the approximated relation: r ~= sqrt(2)*x or r ~= sqrt(2)*y
        # The equation from Qian, et al. (1991) is IN CONTRADICTION with referrenced equation of S. Chandrasekhar (1943)
        # Therefore, the equation from the original (1943) paper will be used!
        sigma = np.sqrt(2.0*self.D*self.time_interval)  # from the Chandrasekhar's paper for random walk along 1D
        # k - because of coordinates x, y - independent and summed in displacement vector r
        k = 1/(np.sqrt(2))
        # k = 1/(2*sigma*np.sqrt(np.pi))  # coefficient for conformity with the equation from Qian, et al. (1991) WRONG
        # print(sigma, k)
        x_step = k*np.random.normal(0, sigma)
        x_step *= self.calibration
        y_step = k*np.random.normal(0, sigma)
        y_step *= self.calibration
        # rounding for trimming unnecessary precision (the values provided in pixels, extra precision could be overkill)
        x_step = np.round(x_step, 3)
        y_step = np.round(y_step, 3)
        self.x_generated_steps.append(x_step)
        self.y_generated_steps.append(y_step)
        r = np.round(np.sqrt(x_step**2 + y_step**2), 3)
        self.r_generated.append(r)
        # print(x_step, y_step, r)
        x_next = self.coordinates[len(self.coordinates)-1][0] + x_step
        y_next = self.coordinates[len(self.coordinates)-1][1] + y_step
        # print(self.coordinates[len(self.coordinates)-1])
        # print(x_next, y_next)
        self.coordinates.append([x_next, y_next])
        return [x_next, y_next]

    def get_statistics(self, size: int = 1000):
        """Plotting the example of generated x,y steps and displacements r."""
        # The equation for 1D random walk from the Chandrasekhar's paper (1943) is used for simulate 1D diffusion step
        sigma = np.sqrt(2.0*self.D*self.time_interval)
        k = 1/(np.sqrt(2))
        if (size > 0) and (size != 1000):
            self.x_steps = np.zeros(size, dtype=float)
            self.y_steps = np.zeros(1000, dtype=float)
            self.r = np.zeros(1000, dtype=float)
        self.x_steps = self.calibration*k*np.random.normal(0, sigma, size)
        self.y_steps = self.calibration*k*np.random.normal(0, sigma, size)
        # Attention: below only additional brackets help to compute the euclidian radius on the arrays at once!
        self.r = np.sqrt((np.power(self.x_steps, 2)) + (np.power(self.y_steps, 2)))
        plt.figure()
        nSteps = 50
        iStep = float(4*sigma/nSteps)
        bins = [-2*sigma + iStep*i for i in range(nSteps+1)]
        # Plotting of histograms for checking of distrubition with defined size
        plt.hist(self.x_steps, bins=bins, density=True, alpha=0.5, label='x')
        plt.hist(self.y_steps, bins=bins, density=True, alpha=0.5, label='y')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.figure()
        bins = [iStep*i for i in range(int(nSteps/2) + 1)]
        # print(bins)
        plt.hist(self.r, bins=bins, density=True, label='r')
        plt.legend(loc='upper right')
        plt.tight_layout()

    def save_generated_stat(self, save_figures: bool = False, save_stats: bool = True):
        plt.figure()
        plt.title("Generated displacements on axes")
        sigma = np.sqrt(2.0*self.D*self.time_interval)
        nSteps = 50
        iStep = float(4*sigma/nSteps)
        bins = [-2*sigma + iStep*i for i in range(nSteps+1)]
        plt.hist(self.x_generated_steps, bins=bins, density=True, alpha=0.5, label='x')
        plt.hist(self.y_generated_steps, bins=bins, density=True, alpha=0.5, label='y')
        plt.legend(loc='upper right')
        plt.tight_layout()
        scriptFolder = os.getcwd()
        default_folder = "tests"
        default_path_for_saving = os.path.join(scriptFolder, default_folder)
        if save_figures:
            default_name = "X,Y histograms.png"
            path = os.path.join(default_path_for_saving, default_name)
            plt.savefig(path, dpi=300)
            plt.close()
        plt.figure()
        plt.title("Generated overall displacement (r)")
        sigma = np.sqrt(2.0*self.D*self.time_interval)
        nSteps = 25
        iStep = float(2*sigma/nSteps)
        bins = [iStep*i for i in range(nSteps+1)]
        plt.hist(self.r_generated, bins=bins, density=True, alpha=0.5, label='r')
        plt.legend(loc='upper right')
        plt.tight_layout()
        if save_figures:
            default_name = "r histogram.png"
            path = os.path.join(default_path_for_saving, default_name)
            plt.savefig(path, dpi=300)
            plt.close()
        # Saving all generated displacements as primitive txt files
        if save_stats:
            default_name = "x.txt"
            # Open and create the txt file if it didn't exist before opening and writing the generated files there
            with open(os.path.join(default_path_for_saving, default_name), 'w') as textfile:
                string_numbers = str(self.x_generated_steps)
                # Cutting out all unnecessary signs
                string_numbers = string_numbers.replace('[', '')
                string_numbers = string_numbers.replace(']', '')
                string_numbers = string_numbers.replace(',', '')
                # print(string_numbers)
                textfile.write(string_numbers)
            default_name = "y.txt"
            # Open and create the txt file if it didn't exist before opening and writing the generated files there
            with open(os.path.join(default_path_for_saving, default_name), 'w') as textfile:
                string_numbers = str(self.y_generated_steps)
                # Cutting out all unnecessary signs
                string_numbers = string_numbers.replace('[', '')
                string_numbers = string_numbers.replace(']', '')
                string_numbers = string_numbers.replace(',', '')
                # print(string_numbers)
                textfile.write(string_numbers)
            default_name = "r.txt"
            # Open and create the txt file if it didn't exist before opening and writing the generated files there
            with open(os.path.join(default_path_for_saving, default_name), 'w') as textfile:
                string_numbers = str(self.r_generated)
                # Cutting out all unnecessary signs
                string_numbers = string_numbers.replace('[', '')
                string_numbers = string_numbers.replace(']', '')
                string_numbers = string_numbers.replace(',', '')
                # print(string_numbers)
                textfile.write(string_numbers)


# %% Testing
if __name__ == '__main__':
    diff = diffusion([0, 0])
    diff.get_next_point()
    diff.get_statistics(5000)
    diff.save_generated_stat()
