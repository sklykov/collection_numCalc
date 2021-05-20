# -*- coding: utf-8 -*-
"""
Experiments with simulation of images of fluorescent beads for evaluation of precision of their localization.
In other words, for generationg ground truth data for tracking / segmentation evaluations.

@author: ssklykov
"""
# %% General imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters  # includes many filters
# import cv2  # importing of OpenCV module (not used so far for simplicity)
# print(cv2.__version__)  # checking the version of installed OpenCV
import os
from skimage.io import imsave
import math


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
    offsets = [int(0), int(0)]  # for storing global offset of a bead image (matrix with its profile)
    # Since the PSF kernel is saved as the class attribute, the collection of following parameters also useful:
    NA = 1.25
    wavelength = 532  # in nanometers
    calibration = 110  # in nanometer/pixel
    max_pixel_value_bead = 255  # for 8bit image

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
            elif image_type == 'float':
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
    def get_bead_img_arbit_center(self, i_offset: float, j_offset: float, max_pixel_val, round_precision: int = 3):
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
        round_precision: int, optional
            Restrict the precision of calculation of float differences to control (estimate) possible rounding errors.
            The default is 3.

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
        self.offsets[0] = int(0)  # reinitilization for repeating generation
        self.offsets[1] = int(0)  # reinitilization for repeating generation

        # Attempt to avoid rounding errors - introducing delta_precision (difference between 0.0 - 0.0 != 0.0 always)
        if round_precision > 1:
            delta_precision = 1/(pow(10, round_precision + 2))
            delta_precision = round(delta_precision, round_precision + 3)
        # print(delta_precision)

        # Using the math module modf() instead of manually defining float and integer part of input float values
        (i_float_part, i_integer_part) = math.modf(i_offset)
        (j_float_part, j_integer_part) = math.modf(j_offset)
        self.offsets[0] = int(i_integer_part)
        self.offsets[1] = int(j_integer_part)
        i_offset = round(i_float_part, round_precision)  # attempt to avoid rounding errors
        j_offset = round(j_float_part, round_precision)  # attempt to avoid rounding errors
        # Calculating the profile of the sub-centralized bead - the bead that is shifted less than 1 pixel in any
        # specified direction - because shifts more than that could be simply drawn on the scene
        i_center = round((float(self.height // 2) + i_offset), round_precision)
        j_center = round((float(self.width // 2) + j_offset), round_precision)
        # print(i_center, j_center)
        if self.bead_type == "even round":
            radius = round(self.character_size*0.5, round_precision + 1)
            for i in range(self.width):
                for j in range(self.height):
                    distance = np.round(np.sqrt(np.power((i - i_center), 2) + np.power((j - j_center), 2)),
                                        round_precision + 1)
                    # print(i, j, ":", distance - radius)
                    # !!! : Always in any calculation with floating points there may be comparison value problems
                    # HACK below: use the comparison with some delta that depends on precision of calculations
                    if ((distance - radius) < (1.0 - delta_precision)):
                        # print(distance-radius)
                        self.bead_img[i, j] = max_pixel_val
                    if ((distance - radius) < (1.0 - delta_precision)) and ((distance - radius) >= 0):
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
            Wavelength of fluorescence emitted by the object (Assumed monochromal case only!).
            The units of wavelength MUST BE in agreement with calibration parameter: nm <=> nm/pixel(!).
        pixel_distance : float.
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
    def show_PSF(max_intensity, size: int, NA: float, wavelength: float, calibration: float, save: bool = False):
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
            Wavelength of fluorescence emitted by the object (Assumed monochromal case only!).
            The units of wavelength MUST BE in agreement with calibration parameter: nm <=> nm/pixel(!).
        calibration : float
            Calibration: um/pixel or nm/pixel for recalculation pixels to the physical values um or nm.
        save: bool, optional
            Saving in the default folder "tests" the calculated PSF accoring to specified properties.

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
        # saving in the default directory for representation
        if save:
            scriptFolder = os.getcwd()
            default_folder = "tests"
            path = os.path.join(scriptFolder, default_folder)
            # print(path)
            base_name = "PSF.png"
            if not os.path.isdir(path):
                os.mkdir(path)
            if os.path.isdir(path):
                # print(path)
                path_for_bead = os.path.join(path, base_name)
                imsave(path_for_bead, img)

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
            Wavelength of fluorescence emitted by the object (Assumed monochromal case only!).
            The units of wavelength MUST BE in agreement with calibration parameter: nm <=> nm/pixel(!).
        calibration : float
            Calibration: um/pixel or nm/pixel for recalculation pixels to the physical values um or nm.

        Raises
        ------
        ValueError
            If some of the specified parameters (NA, wavelength, calibration) is negative!.

        Returns
        -------
        None. Recorder PSF kernel is stored as the class attribute - np.float32 matrix.

        """
        self.NA = NA
        self.wavelength = wavelength
        self.calibration = calibration
        if (NA <= 0.0) or (wavelength <= 0.0) or (calibration <= 0.0):
            raise ValueError("Some of the specified parameters (NA, wavelength, calibration) is negative!")
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
            Wavelength of fluorescence emitted by the object (Assumed monochromal case only!).
            The units of wavelength MUST BE in agreement with calibration parameter: nm <=> nm/pixel(!).
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
            Wavelength of fluorescence emitted by the object (Assumed monochromal case only!).
            The units of wavelength MUST BE in agreement with calibration parameter: nm <=> nm/pixel(!).
        calibration : float
            Calibration: um/pixel or nm/pixel for recalculation pixels to the physical values um or nm.

        Returns
        -------
        None. The image stored in the class' attribute self.bead_img

        """
        # Calculate profile of shifted less than 1 pixel ideal bead (not blurred by convolving with PSF)
        self.get_bead_img_arbit_center(i_offset, j_offset, max_pixel_val)
        # Accounting of diffraction blurring
        self.calc_border_extended_PSF()
        self.max_pixel_value_bead = max_pixel_val
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
            Wavelength of fluorescence emitted by the object (Assumed monochromal case only!).
            The units of wavelength MUST BE in agreement with calibration parameter: nm <=> nm/pixel(!).
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
        # For accounting of shifts in image size in comparison with shifted case
        self.width_centrilized_trimmed_bead = self.width
        self.height_centrilized_trimmed_bead = self.height

    # %% Calculation of centralized and shifted less than 1 pixel blurred bead by convolving its profile with PSF kernel
    def get_whole_shifted_blurred_bead(self, i_offset: float, j_offset: float, max_pixel_val,
                                       NA: float, wavelength: float, calibration: float, round_precision: int = 3):
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
            Wavelength of fluorescence emitted by the object (Assumed monochromal case only!).
            The units of wavelength MUST BE in agreement with calibration parameter: nm <=> nm/pixel(!).
        calibration : float
            Calibration: um/pixel or nm/pixel for recalculation pixels to the physical values um or nm.
        round_precision: int, optional
            Restrict the precision of calculation of float differences to control (estimate) possible rounding errors
            The default is 3.

        Returns
        -------
        None. The image stored in the class' attribute self.bead_img

        """
        # Calculate profile of shifted less than 1 pixel ideal bead (not blurred by convolving with PSF)
        self.get_bead_img_arbit_center(i_offset, j_offset, max_pixel_val)
        # Accounting of diffraction blurring
        self.calculate_img_PSF(NA, wavelength, calibration)
        calibration_sum = np.sum(self.kernel_PSF)
        # Calculate blurred image as whole unblurred image convolved with PSF kernel
        convolved_bead = filters.convolve(np.float32(self.bead_img), self.kernel_PSF, mode='reflect')
        convolved_bead /= calibration_sum  # calibrate to eliminate convolution matrix enhancing signal
        correction = max_pixel_val/np.max(convolved_bead)
        convolved_bead *= correction  # calibrate maximum value to the specified value by user
        self.max_pixel_value_bead = max_pixel_val  # for saving this parameter in the report
        if self.image_type == 'uint8':
            self.bead_img = np.uint8(convolved_bead)
        self.trim_image_for_scene()
        # print(self.height, self.width)

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
            # Changing of width and height attributes of the image below (redundant conversion to fix rounding error)
            self.width = int(j_max + 1 - j_min)
            self.height = int(i_max + 1 - i_min)

    # %% Calculate of origin coordinates
    def get_origin_coordinates(self) -> list:
        """
        Calculates the coordinates of origin (zero pixel) of bead image for further drawing (placing) it on a scene.
        It implies that the image generated and the coordinates of its center already calculated before.

        Returns
        -------
        list
            [i_origin:int, j_origin:int].

        """
        # redundant conversion for int to fix possible rounding error
        i_origin = int(self.offsets[0] - int((self.height-1)//2))
        j_origin = int(self.offsets[1] - int((self.width-1)//2))
        return [i_origin, j_origin]

    # %% Saving of generated bead's image (profile)
    def save_bead_image(self, base_name: str = "001.jpg"):
        """
        Saving the generated image of the bead.

        Parameters
        ----------
        base_name : str, optional
            The base name for an image saving, it should include extension of image. The default is "001.jpg".

        Returns
        -------
        None.

        """
        scriptFolder = os.getcwd()
        default_folder = "tests"
        path = os.path.join(scriptFolder, default_folder)
        # print(path)
        if not os.path.isdir(path):
            os.mkdir(path)
        if os.path.isdir(path):
            # print(path)
            path_for_bead = os.path.join(path, base_name)
            splitted_name = (base_name.split("."))
            extension = splitted_name[len(splitted_name) - 1]  # extract extension from the base name
            # print(extension)
            if extension == "jpg" or extension == "jpeg":
                imsave(path_for_bead, self.bead_img, quality=100)
            else:
                imsave(path_for_bead, self.bead_img)

    # %% Saving the used parameters in the default folder
    def save_used_parameters(self, default_folder: str = "tests"):
        """
        Saving some of the used for generation of the bead sample parameters in some specified folder that
        is placed in the same folder where script is.

        Parameters
        ----------
        default_folder : str, optional
            Name of the folder for saving. The default is "tests".

        Returns
        -------
        None.

        """
        scriptFolder = os.getcwd()
        default_path_for_saving = os.path.join(scriptFolder, default_folder)
        default_name = "parameters.txt"
        with open(os.path.join(default_path_for_saving, default_name), 'a') as textfile:
            string = "Maximum intensity through bead = " + str(self.max_pixel_value_bead) + "\n"
            textfile.write(string)
            string = "NA = " + str(self.NA) + "\n"
            textfile.write(string)
            string = "wavelength = " + str(self.wavelength) + "\n"
            textfile.write(string)
            string = "calibration = " + str(self.calibration) + "\n"
            textfile.write(string)


# %% Testing features
if __name__ == '__main__':
    # Common parameters:
    bead_intensity = 255  # % of maximal value of 255 for 8bit gray image
    wavelength = 486
    NA = 1.2
    calibration = 111
    even_bead = image_beads(character_size=28)  # make the size of a bead bigger for representation
    # even_bead.get_centralized_bead(bead_intensity)
    # image_beads.show_PSF(bead_intensity, 6, NA, wavelength, calibration, save=True)
    # even_bead.calculate_img_PSF(NA, wavelength, calibration)
    # even_bead.get_bead_img_arbit_center(0.0, 0.5, bead_intensity)
    # even_bead.calc_border_extended_PSF()
    # even_bead.get_centralized_blurred_bead(bead_intensity, NA, wavelength, calibration)
    # even_bead.get_whole_centr_blurred_bead(bead_intensity, NA, wavelength, calibration)
    # # even_bead.trim_image_for_scene()
    even_bead.get_whole_shifted_blurred_bead(0.0, 0.1, bead_intensity, NA, wavelength, calibration)
    origin = even_bead.get_origin_coordinates()
    # print(even_bead.height_centrilized_trimmed_bead, even_bead.width_centrilized_trimmed_bead)
    print(even_bead.height, even_bead.width)
    # print(even_bead.height_changed, even_bead.width_changed)
    # even_bead.get_shifted_blurred_bead(0.6, 0.3, bead_intensity, NA, wavelength, calibration)

    even_bead.plot_bead()
    # even_bead.save_bead_image("even_trimmed_shifted_blurred_bead3.png")
