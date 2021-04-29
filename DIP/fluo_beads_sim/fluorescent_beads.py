# -*- coding: utf-8 -*-
"""
Experiments with simulation of images of fluorescent beads

@author: ssklykov
"""
# %% General imports
import numpy as np
import matplotlib.pyplot as plt
# from u_scene import u_scene


# %% class definition
class image_beads():
    # default values
    width = 11
    height = 11
    possible_img_types = ['uint8', 'uint16', 'float']
    character_size = 5
    image_type = 'uint8'
    bead_types = ["even round", "gaussian round", "uneven round"]
    bead_type = "even round"
    bead_img = np.zeros((height, width), dtype=image_type)
    bead_border = np.zeros((height, width), dtype=image_type)

    def __init__(self, image_type: str = 'uint8', character_size: int = 5, bead_type: str = "even round"):
        if image_type in self.possible_img_types:
            self.image_type = image_type
        else:
            self.image_type = 'uint8'
            print("Image type hasn't been recognized, initialized default 8bit gray image")
        if bead_type in self.bead_types:
            self.bead_type = bead_type
        else:
            self.bead_type = "even round"
            print("Bead type hasn't been recognized, initialized default 'even round' type")
        if (character_size != 5) or (image_type != 'uint8'):
            self.character_size = character_size
            self.width = int(character_size*2) + 1
            self.height = int(character_size*2) + 1
            self.bead_img = np.zeros((self.height, self.width), dtype=self.image_type)
            self.bead_border = np.zeros((self.height, self.width), dtype=self.image_type)
            if image_type == 'uint16':
                self.maxPixelValue = 65535
            else:
                self.maxPixelValue = 1.0  # According to the specification of scikit-image

    def get_centrelized_bead(self, max_pixel_val):
        i_center = (self.height // 2)
        j_center = (self.width // 2)
        if self.bead_type == "even round":
            radius = self.character_size*0.5
            for i in range(self.width):
                for j in range(self.height):
                    distance = np.sqrt(np.power((i - i_center), 2) + np.power((j - j_center), 2))
                    # print(i, j, ":", distance - radius)
                    if ((distance - radius) < 1):
                        # print(distance-radius)
                        self.bead_img[i, j] = max_pixel_val
                    if ((distance - radius) < 1) and ((distance - radius) >= 0):
                        # print(distance-radius)
                        self.bead_border[i, j] = max_pixel_val

    def get_bead_img_arbit_center(self, i_center, j_center, max_pixel_val):
        pass

    @staticmethod
    def calculate_PSF(max_intensity, NA: float, wavelength: float, pixel_distance: float, calibration: float) -> float:
        """
        Gaussian approximation of PSF according to the paper of Zhang, et al. (2007)

        Parameters
        ----------
        max_intensity : int or float
            Maximum intensity value in the point source (central pixel).
        NA : float
            NA of a microobjective.
        wavelength : float
            NA of a microobjective.
        pixel_distance : float
            Distance in pixels from the central one (the point source).
        calibration : float
            Calibration: um/pixel or nm/pixel for recalculation pixels to the physical values um or nm.

        Returns
        -------
        float
            Intensity value in the specific pixel.

        """
        max_intensity = float(max_intensity)
        r = calibration*pixel_distance
        sigma = 0.21*wavelength/NA   # Zhang, et al. (2007)
        intensity = max_intensity*np.exp(-((r*r)/(2*sigma*sigma)))
        return intensity

    @staticmethod
    def show_PSF(max_intensity, size: int, NA: float, wavelength: float, calibration: float):
        """
        Plotting calculated with the function 'calculate_PSF' above profile of the approximated PSF.

        Parameters
        ----------
        max_intensity : int or float
            Maximum intensity value in the point source (central pixel).
        size : int
            Size of an image in pixels for plotting.
        NA : float
            NA of a microobjective.
        wavelength : float
            NA of a microobjective.
        calibration : float
            Calibration: um/pixel or nm/pixel for recalculation pixels to the physical values um or nm.

        Returns
        -------
        None as the function. The plot in the external window or console.

        """
        img = np.zeros((size+1, size+1), dtype=float)
        i_center = size//2
        j_center = size//2
        for i in range(size):
            for j in range(size):
                pixel_dist = np.sqrt(np.power((i - i_center), 2) + np.power((j - j_center), 2))
                img[i, j] = image_beads.calculate_PSF(max_intensity, NA, wavelength, pixel_dist, calibration)
                img = np.uint8(img)
        plt.figure()
        plt.imshow(img, cmap=plt.cm.gray, aspect='auto', origin='lower', extent=(0, size, 0, size))
        plt.tight_layout()

    def calc_border_extended_PSF(self):
        pass

    def plot_bead(self):
        # plt.figure()
        # # Below - representation according to the documentation:
        # # plt.cm.gray - for representing gray values, aspect - for filling image values in a window
        # # origin - for adjusting origin of pixels (0, 0), extent - regulation of axis values
        # # extent = (-0.5, numcols-0.5, -0.5, numrows-0.5)) - for origin = 'lower' - documents
        # plt.imshow(self.bead_img, cmap=plt.cm.gray, aspect='auto', origin='lower',
        #            extent=(0, self.width, 0, self.height))
        # plt.tight_layout()
        plt.figure()
        plt.imshow(self.bead_border, cmap=plt.cm.gray, aspect='auto', origin='lower',
                   extent=(0, self.width, 0, self.height))
        plt.tight_layout()


# %% General parameters
width = 1000
height = 1000

# %% Testing features
if __name__ == '__main__':
    even_bead = image_beads(character_size=20)
    even_bead.get_centrelized_bead(200)
    even_bead.plot_bead()
    image_beads.show_PSF(255, 6, 1.25, 532, 110)
