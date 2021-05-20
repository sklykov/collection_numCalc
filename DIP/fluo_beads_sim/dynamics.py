# -*- coding: utf-8 -*-
"""
Simulation of diffusion / motion of microbeads in medium. Calculates only coord

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
    calibration = 1.0  # for compensate diffusion coefficient and time_interval units
    x_steps = np.zeros(1000, dtype=float)
    y_steps = np.zeros(1000, dtype=float)
    r = np.zeros(1000, dtype=float)
    x_generated_steps = []
    y_generated_steps = []
    r_generated = []
    origin = []

    # %% Initialization
    def __init__(self, initial_point: list, D: float = 1.0, time_interval: float = 1.0, calibration: float = 1.0):
        """
        Initialization of the class for generation of 2D diffusion movement.

        Parameters
        ----------
        initial_point : list
            Initial coordinates of a bead in the form [i, j].
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

    # %% Calculate coordinates for the next position
    def get_next_point(self, round_precision: int = 3, debug: bool = False) -> list:
        """
        Calculates the next point in the generated sequence of the diffused bead according to the fundamental
        equation provided in the paper Chandrasekhar (1943) "Stochastic Problems in Physics and Astronomy"

        Parameters
        ----------
        round_precision : int, optional
            The presicion of rounding output coordinates in pixels. The default is 3.
        debug: bool, optional
            Flag for activating some debugging logic below

        Returns
        -------
        list
            Of coordinates [x, y] or [i,j] for the center of diffused bead or whatever coordinates were input.

        """
        # Simulation for each coordinate in the approximated relation: r ~= sqrt(2)*x or r ~= sqrt(2)*y
        # The equation from Qian, et al. (1991) is IN CONTRADICTION with referrenced equation of S. Chandrasekhar (1943)
        # Therefore, the equation from the original (1943) paper will be used!
        sigma = np.sqrt(2.0*self.D*self.time_interval)  # from the Chandrasekhar's paper for random walk along 1D
        # k - because of coordinates x, y - independent and summed in the displacement vector r
        k = 1/(np.sqrt(2))
        # k = 1/(2*sigma*np.sqrt(np.pi))  # coefficient for conformity with the equation from Qian, et al. (1991) WRONG
        # print(sigma, k)
        # NOTE: x and y here stand for i and j respectively in matrix (the scene image) (!)
        x_step = k*np.random.normal(0, sigma)
        x_step *= self.calibration
        y_step = k*np.random.normal(0, sigma)
        y_step *= self.calibration
        # rounding for trimming unnecessary precision (the values provided in pixels, extra precision could be overkill)
        x_step = np.round(x_step, round_precision-1)
        y_step = np.round(y_step, round_precision-1)
        if debug:
            x_step = 0.9999
        self.x_generated_steps.append(x_step)
        self.y_generated_steps.append(y_step)
        r = np.round(np.sqrt(x_step**2 + y_step**2), round_precision)
        self.r_generated.append(r)
        # print(x_step, y_step, r)
        x_next = np.round((self.coordinates[len(self.coordinates)-1][0] + x_step), round_precision)
        y_next = np.round((self.coordinates[len(self.coordinates)-1][1] + y_step), round_precision)
        # print(self.coordinates[len(self.coordinates)-1])
        # print(x_next, y_next)
        self.coordinates.append([x_next, y_next])
        return [x_next, y_next]

    # %% Visualize and check the histograms of x, y, r
    def get_statistics(self, size: int = 1000):
        """
        Plotting the example of generated x,y steps and displacements r as the example of calculations.

        Parameters
        ----------
        size : int, optional
            Number of generated sample points for making graphs (histograms). The default is 1000.

        Returns
        -------
        None.

        """
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
        plt.hist(self.x_steps, bins=bins, density=True, alpha=0.5, label='i')
        plt.hist(self.y_steps, bins=bins, density=True, alpha=0.5, label='j')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.figure()
        bins = [iStep*i for i in range(int(nSteps/2) + 1)]
        # print(bins)
        plt.hist(self.r, bins=bins, density=True, label='r')
        plt.legend(loc='upper right')
        plt.tight_layout()

    # %% Saving the origin of bead image
    def save_bead_origin(self, origin_coordinates: list):
        """
        Created for saving statistics on the calculated origin coordinates for addressing the issue with wrong
        drawing.

        Parameters
        ----------
        origin_coordinates : list
            Calculated origin coordinates.

        Returns
        -------
        None.

        """
        i_origin = origin_coordinates[0]
        j_origin = origin_coordinates[1]
        self.origin.append([i_origin, j_origin])

    # %% Saving of all generated parameters
    def save_generated_stat(self, save_figures: bool = False, save_stats: bool = True, default_folder: str = "tests"):
        """
        Saving the statistics of generated movies: x, y, r displacements and parameters used
        for generation of diffusion.

        Parameters
        ----------
        save_figures : bool, optional
            Save histograms of x, y, r displacements. The default is False.
        save_stats : bool, optional
            Saving raw text files with collected x, y, r, parameters. The default is True.
        default_folder: str, optional
            Default folder for saving the collected statistics, should exist before calling this function in the
            same folder there this script is placed (current working directory).

        Returns
        -------
        None.

        """
        plt.figure()
        plt.title("Generated displacements on axes")
        sigma = np.sqrt(2.0*self.D*self.time_interval)
        nSteps = 50
        iStep = float(4*sigma/nSteps)
        bins = [-2*sigma + iStep*i for i in range(nSteps+1)]
        plt.hist(self.y_generated_steps, bins=bins, alpha=0.5, label='j(x)')
        plt.hist(self.x_generated_steps, bins=bins, alpha=0.5, label='i(y)')
        plt.legend(loc='upper right')
        plt.xlabel("pixels")
        plt.ylabel("counts")
        plt.tight_layout()
        scriptFolder = os.getcwd()
        default_path_for_saving = os.path.join(scriptFolder, default_folder)
        if save_figures:
            default_name = "i(y),j(x) histograms.png"
            path = os.path.join(default_path_for_saving, default_name)
            plt.savefig(path, dpi=300)
            plt.close()
        plt.figure()
        plt.title("Generated overall displacement (r)")
        sigma = np.sqrt(2.0*self.D*self.time_interval)
        nSteps = 25
        iStep = float(2*sigma/nSteps)
        bins = [iStep*i for i in range(nSteps+1)]
        plt.hist(self.r_generated, bins=bins, label='r')
        plt.legend(loc='upper right')
        plt.xlabel("pixels")
        plt.ylabel("counts")
        plt.tight_layout()
        if save_figures:
            default_name = "r histogram.png"
            path = os.path.join(default_path_for_saving, default_name)
            plt.savefig(path, dpi=300)
            plt.close()
        # Saving all generated displacements as primitive txt files
        if save_stats:
            default_name = "i(y).txt"
            # Open and create the txt file if it didn't exist before opening and writing the generated files there
            with open(os.path.join(default_path_for_saving, default_name), 'w') as textfile:
                string_numbers = str(self.x_generated_steps)
                # Cutting out all unnecessary signs
                string_numbers = string_numbers.replace('[', '')
                string_numbers = string_numbers.replace(']', '')
                string_numbers = string_numbers.replace(',', '')
                # print(string_numbers)
                textfile.write(string_numbers)
            default_name = "j(x).txt"
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
            default_name = "parameters.txt"
            with open(os.path.join(default_path_for_saving, default_name), 'a') as textfile:
                string = "D = " + str(self.D) + "\n"
                textfile.write(string)
                string = "Time interval = " + str(self.time_interval) + "\n"
                textfile.write(string)
                string = "Calibration parameter = " + str(self.calibration) + "\n"
                textfile.write(string)
                string = "sigma = " + str(np.round(sigma, 3)) + "\n"
                textfile.write(string)
                # textfile.write("\n")
            default_name = "i, j rounded coordinates.txt"
            with open(os.path.join(default_path_for_saving, default_name), 'w') as textfile:
                for coordinate in self.coordinates:
                    string = str(coordinate[0]) + " " + str(coordinate[1]) + "\n"  # whitespace-delimited recording
                    textfile.write(string)
            default_name = "origin coordinates.txt"
            if len(self.origin) > 0:
                with open(os.path.join(default_path_for_saving, default_name), 'w') as textfile:
                    for coordinate in self.origin:
                        string = str(coordinate[0]) + " " + str(coordinate[1]) + "\n"  # whitespace-delimited recording
                        textfile.write(string)


# %% Testing
if __name__ == '__main__':
    diff = diffusion([0, 0])
    diff.get_next_point()
    diff.get_statistics(5000)
    diff.save_generated_stat()
