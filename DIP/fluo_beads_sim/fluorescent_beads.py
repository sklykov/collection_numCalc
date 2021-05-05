# -*- coding: utf-8 -*-
"""
Experiments with simulation of images of fluorescent beads for evaluation of precision of their localization.

@author: ssklykov
"""
# %% General imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters  # includes many filters
# import cv2  # importing of OpenCV module (not used so far for simplicity)
# print(cv2.__version__)  # checking the version of installed OpenCV


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
    bead_img = np.zeros((height, width), dtype=image_type)  # intensity profile of a bead
    bead_border = np.zeros((height, width), dtype=image_type)  # edge or border of a bead
    maxPixelValue = 255  # default for a 8bit gray image
    kernel_PSF = []  # for storing the kernel for convolution ("diffraction blurring of sharp edges")
    bead_conv_border = []  # for storing blurred border
    offsets = [0, 0]  # for storing global offset of a bead image (matrix with its profile)

    # %% Constructor
    def __init__(self, image_type: str = 'uint8', character_size: int = 5, bead_type: str = "even round"):
        """
        Generate basis class with the image (representation) of fluorescent bead with various intensity profile.

        Parameters
        ----------
        image_type : str, optional
            Image type: 8bit, 16bit or float gray image. The default is 'uint8'.
        character_size : int, optional
            Characteristic size of a bead. For 'even round' bead - radius. The default is 5.
        bead_type : str, optional
            Type of fluorescent bead profile: 'even round', 'gaussian round', 'uneven round'.
            The default is "even round".

        Returns
        -------
        None.

        """
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

    # %% Generate centralized profile
    def get_centralized_bead(self, max_pixel_val):
        """
        Generate centared image of a bead with specified type.

        Parameters
        ----------
        max_pixel_val : int(uint8 or uint16) or float
            Maximum intensity value in the center of a bead (or just through a bead intensity profile).

        Returns
        -------
        None. Calculated 2D image and the border stored in class' attributes self.bead_img and self.bead_border.

        """
        # Below - for repeating generation of initial sized bead after trimming of the previously generated image
        if self.bead_type == "even round":
            self.height = int(self.character_size*2) + 1
            self.width = int(self.character_size*2) + 1
            self.bead_img = np.zeros((self.height, self.width), dtype=self.image_type)
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

    # %% Generate centralized with shift less 1 pixel profile
    def get_bead_img_arbit_center(self, i_offset: float, j_offset: float, max_pixel_val):
        """
        Generate shifted for less than 1 pixel centared before image of a bead with specified type.
        Such splitting on even pixel shift and less than 1 pixel is performed because of even offset
        could be applied only for integration the bead image on the larger scene there even shifts are
        simply drawn by shifting entire bead image.

        Parameters
        ----------
        i_offset : float
            Arbitrary shift that will splitted on even pixel shift stored in self.offsets and less than 1 pixel
            offset applied for an image calculation.
        j_offset : float
            Same as i_offset but in other direction.
        max_pixel_val :  int(uint8 or uint16) or float
            Maximum intensity value in the center of a bead (or just through a bead intensity profile).

        Returns
        -------
        None. Calculated 2D image and the border stored in class' attributes self.bead_img and self.bead_border

        """
        # Below - for repeating generation of initial sized bead after trimming of the previously generated image
        if self.bead_type == "even round":
            self.height = int(self.character_size*2) + 1
            self.width = int(self.character_size*2) + 1
            self.bead_img = np.zeros((self.height, self.width), dtype=self.image_type)
        # The self.offsets will store even pixel offsets for bead for its further introducing to the scene (background)
        self.offsets[0] = 0  # reinitilization
        self.offsets[1] = 0  # reinitilization
        while i_offset >= 1.0:
            self.offsets[0] += 1
            i_offset -= 1.0
        while j_offset >= 1.0:
            self.offsets[1] += 1
            j_offset -= 1.0
        # print(i_offset, j_offset)
        # Calculating the profile of the sub-centralized bead - the bead that is shifted less than 1 pixel in any
        # specified direction - because shifts more than that could be simply drawn on the scene
        i_center = (self.height // 2) + i_offset
        j_center = (self.width // 2) + j_offset
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

    # %% Calculation of PSF for any point depending on pixel distance to the central point
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

    # %% Plotting centralized profile of the calculated PSF
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
        plt.imshow(img, cmap=plt.cm.Spectral, aspect='auto', origin='lower', extent=(0, size, 0, size))
        plt.tight_layout()

    # %% Calculate PSF as the kernel for further convolution
    def calculate_img_PSF(self, NA: float, wavelength: float, calibration: float):
        """
        Calculation PSF as kernel for convolution with automatically calculated size N (NxN convolution kernel).
        PSF kernel is used for calculation of edges blurring due to diffraction.

        Parameters
        ----------
        NA : float
            NA of a microobjective.
        wavelength : float
            NA of a microobjective.
        calibration : float
            Calibration: um/pixel or nm/pixel for recalculation pixels to the physical values um or nm.

        Returns
        -------
        None. Recorder PSF kernel is stored as the class attribute - np.float32 matrix.

        """
        if self.image_type == 'uint8':
            min_pixel = 1  # the smallest meaningful pixel value for accounting of blurring
            dimension = 0
            intensity = self.maxPixelValue
            while intensity >= min_pixel:
                dimension += 1
                intensity = image_beads.calculate_PSF(self.maxPixelValue, NA, dimension, wavelength, calibration)
                intensity = np.uint8(intensity)
            dimension = 2*dimension + 1
            kernel = np.zeros((dimension, dimension), dtype=float)
            i_center = dimension // 2
            j_center = dimension // 2
            for i in range(dimension):
                for j in range(dimension):
                    pixel_dist = np.sqrt(np.power((i - i_center), 2) + np.power((j - j_center), 2))
                    kernel[i, j] = image_beads.calculate_PSF(1, NA, wavelength, pixel_dist, calibration)
            self.kernel_PSF = np.float32(kernel)
            # print(self.kernel_PSF)

    # %% Checking separately how the border experiences the blurring by the calculated PSF application
    def calc_border_extended_PSF(self):
        """
        Calculation of blurred bead's border due to the diffraction blurring of sharp edges.

        Raises
        ------
        ValueError
            If self.bead_border hasn't been calculated before calling this function.

        Returns
        -------
        None. The blurred border is stored in the class' attrubute self.bead_conv_border.

        """
        if np.max(self.bead_border) <= 0:
            raise ValueError("Most probably, there is the zero (empty) image of the bead")

        # Convolution using standard numpy function
        convolved_borders = filters.convolve(np.float32(self.bead_border), self.kernel_PSF, mode='reflect')

        # Convolution using OpenCV function
        # "-1" parameter means the format of the output result (as I understand)
        # convolved_borders = cv2.filter2D(np.float32(self.bead_border), -1, self.kernel_PSF)

        # convolved_borders - using np.float32 format of pixel values
        calibration_sum = np.sum(self.kernel_PSF)
        # print(calibration_sum)
        convolved_borders /= calibration_sum  # calibration because of positive sum of convolution matrix
        if self.image_type == 'uint8':
            self.bead_conv_border = np.uint8(convolved_borders)
        # print(convolved_borders)

    # %% Get the centralized blurred bead's profile by adding blurred border
    def get_centralized_blurred_bead(self, max_pixel_val, NA: float, wavelength: float, calibration: float):
        """
        Calculation of the intensity profile of the specified bead type with blurred edge due to the diffraction.

        Parameters
        ----------
        max_pixel_val : int (uint8 or uint16) or float.
            Maximum intensity value in the center of a bead (or just through a bead intensity profile).
        NA : float
            NA of a microobjective.
        wavelength : float
            NA of a microobjective.
        calibration : float
            Calibration: um/pixel or nm/pixel for recalculation pixels to the physical values um or nm.

        Returns
        -------
        None. Centralized and blurred bead is stored in the class' attribute self.bead_img.

        """
        self.get_centralized_bead(max_pixel_val)  # calculate unblurred centrelized bead
        self.calculate_img_PSF(NA, wavelength, calibration)
        self.calc_border_extended_PSF()
        border_coordinates = []
        for i in range(self.height):
            for j in range(self.width):
                if self.bead_border[i, j] > 0:
                    border_coordinates.append([i, j])
        # print(border_coordinates)
        for border_pair in border_coordinates:
            [i, j] = border_pair
            # increasing intensity on the bead border due to influence of inner bead surface
            # the intensity value increased in 1.5 times in assumption that inner part contributes to the border
            # intensity as well
            if self.image_type == 'uint8':
                self.bead_conv_border[i, j] = np.uint8(np.float32(self.bead_conv_border[i, j])*1.5)
            self.bead_img[i, j] = self.bead_conv_border[i, j]
        for i in range(self.height):
            for j in range(self.width):
                if self.image_type == 'uint8':
                    if self.bead_img[i, j] == 0 and self.bead_conv_border[i, j] > 0:
                        self.bead_img[i, j] = self.bead_conv_border[i, j]

    # %% Get the centralized and shifted less than 1 pixel blurred bead's profile by adding blurred border
    def get_shifted_blurred_bead(self, i_offset: float, j_offset: float, max_pixel_val,
                                 NA: float, wavelength: float, calibration: float):
        """
        Composing the blurring of sub-centrilized image.

        Parameters
        ----------
        i_offset : float
            Arbitrary shift that will splitted on even pixel shift stored in self.offsets and less than 1 pixel
            offset applied for an image calculation.
        j_offset : float
            Same as i_offset but in other direction.
        max_pixel_val :  int(uint8 or uint16) or float
            Maximum intensity value in the center of a bead (or just through a bead intensity profile).
        NA : float
            NA of a microobjective.
        wavelength : float
            NA of a microobjective.
        calibration : float
            Calibration: um/pixel or nm/pixel for recalculation pixels to the physical values um or nm.

        Returns
        -------
        None. The image stored in the class' attribute self.bead_img

        """
        self.get_bead_img_arbit_center(i_offset, j_offset, max_pixel_val)  # calculate unblurred centrelized bead
        self.calculate_img_PSF(NA, wavelength, calibration)
        self.calc_border_extended_PSF()
        border_coordinates = []
        for i in range(self.height):
            for j in range(self.width):
                if self.bead_border[i, j] > 0:
                    border_coordinates.append([i, j])
        # print(border_coordinates)
        for border_pair in border_coordinates:
            [i, j] = border_pair
            # increasing intensity on the bead border due to influence of inner bead surface
            # the intensity value increased in 1.5 times in assumption that inner part contributes to the border
            # intensity as well
            if self.image_type == 'uint8':
                self.bead_conv_border[i, j] = np.uint8(np.float32(self.bead_conv_border[i, j])*1.5)
            self.bead_img[i, j] = self.bead_conv_border[i, j]
        for i in range(self.height):
            for j in range(self.width):
                if self.image_type == 'uint8':
                    if self.bead_img[i, j] == 0 and self.bead_conv_border[i, j] > 0:
                        self.bead_img[i, j] = self.bead_conv_border[i, j]

    # %% Calculation of centralized blurred bead by convolving its profile with PSF kernel
    def get_whole_centr_blurred_bead(self, max_pixel_val, NA: float, wavelength: float, calibration: float):
        """
        Calculate the blurred, centralized image of the specified bead.

        Parameters
        ----------
        max_pixel_val : int (uint8 or uint16) or float.
            Maximum intensity value in the center of a bead (or just through a bead intensity profile).
        NA : float
            NA of a microobjective.
        wavelength : float
            NA of a microobjective.
        calibration : float
            Calibration: um/pixel or nm/pixel for recalculation pixels to the physical values um or nm.

        Returns
        -------
        None. The calculated image of a bead stored in class' attribute self.bead_img

        """
        self.get_centralized_bead(max_pixel_val)  # calculate unblurred centrelized bead
        self.calculate_img_PSF(NA, wavelength, calibration)
        calibration_sum = np.sum(self.kernel_PSF)
        # Calculate blurred image as whole ublurred image convolved with PSF kernel
        convolved_bead = filters.convolve(np.float32(self.bead_img), self.kernel_PSF, mode='reflect')
        convolved_bead /= calibration_sum  # calibrate to eliminate convolution matrix enhancing signal
        correction = max_pixel_val/np.max(convolved_bead)
        convolved_bead *= correction  # calibrate maximum value to the specified value by user
        if self.image_type == 'uint8':
            self.bead_img = np.uint8(convolved_bead)
        self.trim_image_for_scene()

    # %% Calculation of centralized and shifted less than 1 pixel blurred bead by convolving its profile with PSF kernel
    def get_whole_shifted_blurred_bead(self, i_offset: float, j_offset: float, max_pixel_val,
                                       NA: float, wavelength: float, calibration: float):
        """
        Calculate the blurred, shifted to less than 1 pixel from the center image of the specified bead.

        Parameters
        ----------
        i_offset : float
            Arbitrary shift that will splitted on even pixel shift stored in self.offsets and less than 1 pixel
            offset applied for an image calculation.
        j_offset : float
            Same as i_offset but in other direction.
        max_pixel_val :  int(uint8 or uint16) or float
            Maximum intensity value in the center of a bead (or just through a bead intensity profile).
        NA : float
            NA of a microobjective.
        wavelength : float
            NA of a microobjective.
        calibration : float
            Calibration: um/pixel or nm/pixel for recalculation pixels to the physical values um or nm.

        Returns
        -------
        None. The image stored in the class' attribute self.bead_img

        """
        self.get_bead_img_arbit_center(i_offset, j_offset, max_pixel_val)  # calculate unblurred centrelized bead
        self.calculate_img_PSF(NA, wavelength, calibration)
        calibration_sum = np.sum(self.kernel_PSF)
        # Calculate blurred image as whole ublurred image convolved with PSF kernel
        convolved_bead = filters.convolve(np.float32(self.bead_img), self.kernel_PSF, mode='reflect')
        convolved_bead /= calibration_sum  # calibrate to eliminate convolution matrix enhancing signal
        correction = max_pixel_val/np.max(convolved_bead)
        convolved_bead *= correction  # calibrate maximum value to the specified value by user
        if self.image_type == 'uint8':
            self.bead_img = np.uint8(convolved_bead)
        self.trim_image_for_scene()

    # %% Plotting the bead's image or profile
    def plot_bead(self):
        """
        Plotting various class' matrix attributes (specified in the code) as images opened as matplotlib windows.

        Returns
        -------
        None.

        """
        plt.figure()
        # Below - representation according to the documentation:
        # plt.cm.gray - for representing gray values, aspect - for filling image values in a window
        # origin - for adjusting origin of pixels (0, 0), extent - regulation of axis values
        # extent = (-0.5, numcols-0.5, -0.5, numrows-0.5)) - for origin = 'lower' - documents
        plt.imshow(self.bead_img, cmap=plt.cm.gray, aspect='auto', origin='lower',
                   extent=(0, self.width, 0, self.height))
        plt.tight_layout()
        if np.size(self.bead_conv_border) > 0:
            plt.figure()
            plt.imshow(self.bead_conv_border, cmap=plt.cm.gray, aspect='auto', origin='lower',
                       extent=(0, self.width, 0, self.height))
            plt.tight_layout()

    # %% Cutting out zero background from bead's image or profile
    def trim_image_for_scene(self):
        """
        Trimming (cutting out) the background (zero pixels) from the bead image.

        Returns
        -------
        None. It works on the bead image - class attribute self.bead_img.

        """
        if self.image_type == 'uint8':
            min_pixel = 1  # minimal meaningful pixel value for this type of image
            i_min = self.height
            j_min = self.width
            i_max = 0
            j_max = 0
            for i in range(self.height):
                for j in range(self.width):
                    if self.bead_img[i, j] >= min_pixel:
                        if i < i_min:
                            i_min = i
                        if j < j_min:
                            j_min = j
                        if i > i_max:
                            i_max = i
                        if j > j_max:
                            j_max = j
            # print(i_min, j_min, i_max, j_max)  # debugging
            self.bead_img = self.bead_img[i_min:i_max+1, j_min:j_max+1]
            self.width = j_max + 1 - j_min
            self.height = i_max + 1 - i_min


# %% Testing features
if __name__ == '__main__':
    even_bead = image_beads(character_size=20)
    # even_bead.get_centralized_bead(200)
    # even_bead.plot_bead()
    # image_beads.show_PSF(255, 6, 1.25, 532, 110)
    # even_bead.calculate_img_PSF(1.25, 532, 110)
    # even_bead.calc_border_extended_PSF()
    # even_bead.plot_bead()
    # even_bead.get_centralized_blurred_bead(200, 1.25, 532, 110)
    # even_bead.plot_bead()

    even_bead.get_whole_centr_blurred_bead(200, 1.25, 532, 110)
    # even_bead.trim_image_for_scene()
    even_bead.plot_bead()

    even_bead.get_whole_shifted_blurred_bead(0.6, 0.3, 200, 1.25, 532, 110)
    even_bead.plot_bead()
    # even_bead.get_shifted_blurred_bead(0.6, 0.3, 200, 1.25, 532, 110)
    # even_bead.plot_bead()
