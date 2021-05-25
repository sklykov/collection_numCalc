# -*- coding: utf-8 -*-
"""
Container for building a scene with fluorescent objects (i.e., scene plays a role of background or frame).

@author: ssklykov
"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
# from skimage.util import img_as_ubyte
import os
from skimage.io import imsave
from scipy.ndimage import measurements


# %% class definition
class u_scene():
    """Class composing all capabilities of building image (numpy 2D array) with some objects drawn on the scene.
    The image commonly is designated as width x height (e.g., 800x600)"""

    # default values
    width = 100
    height = 100
    possible_img_types = ['uint8', 'uint16', 'float']
    image_type = 'uint8'
    scene_image = np.zeros((height, width), dtype=image_type)
    maxPixelValue = 255
    counter = 1  # counting how many images saved along generation
    centers_of_mass = []

    # %% Constructor
    def __init__(self, width: int, height: int, image_type: str = 'uint8'):
        """
        Initialize the blank (dark) scene image with the specified type (800x600 8bit image as an example)

        Parameters
        ----------
        width : int
            Width of the initialized image (scene)
        height : int
            Height of the initialized image (scene)
        image_type : str, optional
            Image type used for pixel value calculations. Possible values are: 'uint8', 'uint16', 'float'.
            The default is 'uint8'.

        Returns
        -------
        None.

        """
        width = abs(width)
        height = abs(height)
        if width > 0:
            self.width = width
        if height > 0:
            self.width = width
        if image_type in self.possible_img_types:
            self.image_type = image_type
        else:
            self.image_type = 'uint8'
            print("Image type hasn't been recognized, initialized default 8bit gray image")
        if (width != 100) or (height != 100) or (image_type != 'uint8'):
            # non default values => re-initialization of the class attributes
            self.scene_image = np.zeros((height, width), dtype=self.image_type)
            self.width = width
            self.height = height
            if self.image_type == 'uint16':
                self.maxPixelValue = 65535
            elif self.image_type == 'float':
                self.maxPixelValue = 1.0  # According to the specification of scikit-image

    # %% Supportive functions
    def cast_pixels_sum(self, pixels_sum):
        """
        Casting of input result of pixel summing to conform with data type of the used image.

        Parameters
        ----------
        pixels_sum : uint8, uint16 or float
            Sum of pixels (mask + scene (background)).

        Returns
        -------
        value_returned : uint8, uint16 or float
            Returns casted / corrected pixel value.

        """
        if (pixels_sum) <= self.maxPixelValue:
            # additional conversion for insuring of conformity with data type
            if self.image_type == 'uint8':
                value_returned = np.uint8(pixels_sum)
            elif self.image_type == 'uint16':
                value_returned = np.uint16(pixels_sum)
            else:
                value_returned = float(pixels_sum)
        else:
            value_returned = self.maxPixelValue
        return value_returned

    def get_j_finish(self, j_start: int, nCols: int) -> int:
        """
        Calculation of maximum j index for adding mask, preventing it to be for out of bounds.

        Parameters
        ----------
        j_start : int
            Starting index for filling mask in.
        nCols : int
            Number of columns in mask that should be added to the scene.

        Returns
        -------
        int
            Ending ("final") index j for filling mask into the scene.

        """
        if ((j_start + nCols) < self.width):  # checking that starting/ending of summing are not out of bounds
            j_finish = j_start + nCols
        else:
            j_finish = self.width
        return j_finish

    def get_i_finish(self, i_start: int, nRows: int) -> int:
        """
        Calculation of maximum i index for adding mask, preventing it to be for out of bounds

        Parameters
        ----------
        i_start : int
             Starting index for filling mask in.
        nRows : int
            Number of columns in mask that should be added to the scene.

        Returns
        -------
        int
            Ending ("final") index j for filling mask into the scene.

        """
        if ((i_start + nRows) < self.height):  # checking that starting/ending of summing are not out of bounds
            i_finish = i_start + nRows
        else:
            i_finish = self.height
        return i_finish

    # %% Drawing of an object with some intensity mask (profile)
    def add_mask(self, i_start: int, j_start: int, mask, debug: bool = False):
        """
        Adding the "mask" - representation of the object (basically, less than the scene (background) image).
        Contradictory, i_start represents "y" coordinate, j_start - "x", due to array representation of column and row.
        This function accepts coordinates of image origin - starting pixel for drawing (like zero pixel).
        The coordinates (j_start, i_start) as (x, y) could be negative or exceeding the scene sizes - in such case
        whenever it possible, only the part of an object image will be added.

        Parameters
        ----------
        i_start : int
            Start pixel (y coordinate) for drawing of an image ("mask").
        j_start : int
            Start pixel (x coordinate) for drawing of an image ("mask").
        mask : np.array
            2D np.array ("mask") with pixel values which represent the object.
        debug: bool, optional
            Flag for saving some internal statistical values for checking of possible bugs during calculations.
            The default is False.

        Returns
        -------
        None.
        The scene collected as internal attribute of this class.

        """
        (nRows, nCols) = np.shape(mask)  # getting of sizes of mask

        # Below is checking that the mask is not empty, it should be 1x1 matrix at least
        if (nRows == 0) or (nCols == 0):
            raise(IndexError('Provided mask is empty along some of its axis'))

        # Below is checking that the i_start and j_start makes sense to apply to the scene image:
        # i_start and j_start could be negative, but at least 1 point should be added to a scene
        # also, j associates with WIDTH, so with # of columns! i - with rows!
        if ((i_start + nRows) < 1) or ((j_start + nCols) < 1):
            raise(IndexError('Provided i_start or j_start is not conformed with the mask sizes'))

        # Below is checking filling parameters (i_start, j_start) is laying on an scene image
        if (i_start >= self.height) or (j_start >= self.width):
            raise(IndexError("Starting indices for adding mask is out of scene image bounds"))

        # i_start, j_start > 0 both, filling some mask into a scene image - basic check for conformity
        if (i_start >= 0) and (j_start >= 0) and (nRows > 0) and (nCols > 0):
            # Attempt to speed up the adding mask to a scene: transferring pixel values as chunk with rows
            if ((i_start + nRows) < self.height):  # checking that fast sum over y axis could be performed
                i_finish = i_start + nRows
                j_finish = self.get_j_finish(j_start, nCols)

                # "Fast summing" - adding the whole rows (all columns) to the image (scene)
                for j in range(j_start, j_finish):  # summing along j axis
                    # checking the conformity with image type
                    if np.max(self.scene_image[i_start:i_finish, j] + mask[:, j-j_start]) <= self.maxPixelValue:
                        self.scene_image[i_start:i_finish, j] += mask[:, j-j_start]  # fast adding mask to a scene
                    else:
                        # checking each pixel from a scene and added from a mask pixel to be in range with image type
                        for i in range(i_start, i_finish):
                            pixels_sum = self.scene_image[i, j] + mask[i-i_start, j-j_start]
                            self.scene_image[i, j] = self.cast_pixels_sum(pixels_sum)

            # Attempt to speed up the adding mask to a scene: transferring pixel values as a chunk with columns
            elif ((j_start + nCols) < self.width):  # checking that fast sum over i axis could be performed
                j_finish = j_start + nCols
                i_finish = self.get_i_finish(i_start, nRows)

                # "Fast summing" - along column - adding all rows at once
                for i in range(i_start, i_finish):  # summing along j axis
                    # checking the conformity with image type
                    if np.max(self.scene_image[i, j_start:j_finish] + mask[i-i_start, :]) <= self.maxPixelValue:
                        self.scene_image[i, j_start:j_finish] += mask[i-i_start, :]  # fast adding mask to a scene
                    else:
                        # checking each pixel from a scene and added from a mask pixel to be in range with image type
                        for j in range(j_start, j_finish):
                            pixels_sum = self.scene_image[i, j] + mask[i-i_start, j-j_start]
                            self.scene_image[i, j] = self.cast_pixels_sum(pixels_sum)

            # filling right upper corner with exceptional case - when mask is out of image bounds
            else:
                i_finish = self.height
                j_finish = self.width
                for i in range(i_start, i_finish):
                    for j in range(j_start, j_finish):
                        pixels_sum = self.scene_image[i, j] + mask[i-i_start, j-j_start]
                        self.scene_image[i, j] = self.cast_pixels_sum(pixels_sum)

        # Making correction of i_start, j_start if some of them is negative for working with partial mask overlap
        if (i_start < 0) or (j_start < 0):
            i_mask_start = 0
            j_mask_start = 0
            if (i_start < 0):
                nRows += i_start  # it will draw the mask if it partially overlaps with image boundaries
                i_mask_start = abs(i_start)
                i_start = 0
            if (j_start < 0):
                nCols += j_start
                j_mask_start = abs(j_start)
                j_start = 0
            i_finish = self.get_i_finish(i_start, nRows)
            j_finish = self.get_j_finish(j_start, nCols)
            for i in range(i_start, i_finish):
                for j in range(j_start, j_finish):
                    pixels_sum = self.scene_image[i, j] + mask[i+i_mask_start, j+j_mask_start]
                    self.scene_image[i, j] = self.cast_pixels_sum(pixels_sum)
        # HINT: below is controlling of simulation - calculation of center of mass of added mask (generated scene)
        if debug:
            (i_mass_center, j_mass_center) = measurements.center_of_mass(self.scene_image)
            self.centers_of_mass.append([i_mass_center, j_mass_center])
        # print([i_mass_center, j_mass_center])

    # %% Plotting the summurized image (scene) with all objects
    def plot_image(self):
        """
        Plotting the self.scene composed with added masks (objects) / noise.

        Returns
        -------
        Plotted the scene (picture) on the separate window using matplotlib library

        """
        plt.figure()
        # Below - representation according to the documentation:
        # plt.cm.gray - for representing gray values, aspect - for filling image values in a window
        # origin - for adjusting origin of pixels (0, 0), extent - regulation of axis values
        # extent = (-0.5, numcols-0.5, -0.5, numrows-0.5)) - for origin = 'lower' - documents
        plt.imshow(self.scene_image, cmap=plt.cm.gray, aspect='auto', origin='lower',
                   extent=(0, self.width, 0, self.height))
        plt.tight_layout()

    # %% Clearing the scene
    def clear_scene(self):
        """
        Clearing the scene (background) image by re-initialize it to zero values (completely dark).

        Returns
        -------
        None.

        """
        self.scene_image = np.zeros((self.height, self.width), dtype=self.image_type)

    # %% Saving generated scene image
    def save_scene(self, base_extension: str = "jpg"):
        """
        Saving the scene (image) with all collected masks (objects) on it.

        Parameters
        ----------
        base_extension : str, optional
            The base extension for saving images (like jpg, png, tiff, etc). The default is "jpg".

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
            base_name = str(self.counter) + "." + base_extension
            self.counter += 1
            path_for_bead = os.path.join(path, base_name)
            if base_extension == "jpg" or base_extension == "jpeg":
                imsave(path_for_bead, self.scene_image, quality=100)
            else:
                imsave(path_for_bead, self.scene_image)


# %% Testing class methods / construction
if __name__ == '__main__':
    uScene = u_scene(150, 150, 'uint8')
    mask = np.ones((20, 20), dtype='uint8')
    mask = mask[:, :]*256
    uScene.add_mask(40, 40, mask)
    uScene.add_mask(80, 80, mask)
    uScene.plot_image()
    uScene.save_scene()
