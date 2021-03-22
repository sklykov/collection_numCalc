# -*- coding: utf-8 -*-
"""
Container for building a scene with fluorescent objects

@author: ssklykov
"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
# from skimage.util import img_as_ubyte


# %% class definition
class u_scene():

    # default values
    width = 100
    height = 100
    possible_img_types = ['uint8', 'uint16', 'float']
    image_type = 'uint8'
    scene_image = np.zeros((width, height), dtype=image_type)
    maxPixelValue = 255

    def __init__(self, width: int, height: int, image_type: str = 'uint8'):
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
            print("Image type isn't recognized, initialized default 8bit gray image")
        if (width != 100) or (height != 100) or (image_type != 'uint8'):
            self.scene_image = np.zeros((width, height), dtype=self.image_type)
            if image_type == 'uint16':
                self.maxPixelValue = 65535
            else:
                self.maxPixelValue = 1.0  # According to the specification in scikit-image

    def add_mask(self, x_start: int, y_start: int, mask):
        (nRows, nCols) = np.shape(mask)  # getting of sizes of mask
        # print(nRows, nCols, " - sizes of input mask")  # Checking the proper working

        # Below is checking that the mask is not empty, it should be 1x1 matrix at least
        if (nRows == 0) or (nCols == 0):
            raise(IndexError('Provided mask is empty along some of its axis'))

        # Below is checking that the x_start and y_start makes sense to apply to the scene image:
        # x_start and y_start could be negative, but at least 1 point should be added to a scene
        if ((x_start + nRows) < 1) or ((y_start + nCols) < 1):
            raise(IndexError('Provided x_start or y_start is not conformed with mask sizes'))

        # Below is checking filling parameters (x_start, y_start) is laying on an scene image
        if (x_start >= self.height) or (y_start >= self.width):
            raise(IndexError("Starting indices for adding mask is out of scene image bounds"))

        # TODO
        # 1) x_start, y_start - could be negative, make the logic for it - fiiling out of bounds of an image

        # x_start, y_start > 0 both, filling some mask into a scene image
        if (x_start >= 0) and (y_start >= 0):
            # Attempt to speed up the adding mask to a scene: transferring pixel values as chunk with rows
            if ((x_start + nRows) < self.height):  # checking that fast sum over x axis could be performed
                x_finish = x_start + nRows
                if ((y_start + nCols) < self.width):  # checking that starting and ending of summing are not of bounds
                    y_finish = y_start + nCols
                else:
                    y_finish = self.width
                for j in range(y_start, y_finish):  # summing along y axis
                    # checking the conformity with image type
                    if np.max(self.scene_image[x_start:x_finish, j] + mask[:, j-y_start]) <= self.maxPixelValue:
                        self.scene_image[x_start:x_finish, j] += mask[:, j-y_start]  # fast adding mask to a scene
                    else:
                        # checking each pixel from a scene and added from a mask pixel to be in range with image type
                        for i in range(x_start, x_finish):
                            pixels_sum = self.scene_image[i, j] + mask[i-x_start, j-y_start]
                            if (pixels_sum) <= self.maxPixelValue:
                                if self.image_type == 'uint8':
                                    self.scene_image[i, j] = np.uint8(pixels_sum)  # additional conversion
                            else:
                                self.scene_image[i, j] = self.maxPixelValue

            # Attempt to speed up the adding mask to a scene: transferring pixel values as chunk with columns
            elif ((y_start + nCols) < self.width):  # checking that fast sum over x axis could be performed
                y_finish = y_start + nCols
                if ((x_start + nRows) < self.height):  # checking that starting and ending of summing are not of bounds
                    x_finish = x_start + nRows
                else:
                    x_finish = self.height
                for i in range(x_start, x_finish):  # summing along y axis
                    # checking the conformity with image type
                    if np.max(self.scene_image[i, y_start:y_finish] + mask[i-x_start, :]) <= self.maxPixelValue:
                        self.scene_image[i, y_start:y_finish] += mask[i-x_start, :]  # fast adding mask to a scene
                    else:
                        # checking each pixel from a scene and added from a mask pixel to be in range with image type
                        for j in range(y_start, y_finish):
                            pixels_sum = self.scene_image[i, j] + mask[i-x_start, j-y_start]
                            if (pixels_sum) <= self.maxPixelValue:
                                if self.image_type == 'uint8':
                                    self.scene_image[i, j] = np.uint8(pixels_sum)  # additional conversion
                            else:
                                self.scene_image[i, j] = self.maxPixelValue

            # filling right upper corner with exceptional case - when mask is out of image bounds
            else:
                x_finish = self.height
                y_finish = self.width
                for i in range(x_start, x_finish):
                    for j in range(y_start, y_finish):
                        pixels_sum = self.scene_image[i, j] + mask[i-x_start, j-y_start]
                        if (pixels_sum) <= self.maxPixelValue:
                            if self.image_type == 'uint8':
                                self.scene_image[i, j] = np.uint8(pixels_sum)  # additional conversion
                        else:
                            self.scene_image[i, j] = self.maxPixelValue

    def plot_image(self):
        """
        Plotting the self.scene composed with added masks / noise.

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
                   extent=(0, self.height, 0, self.width))
        plt.tight_layout()


# %% Testing class methods / construction
if __name__ == '__main__':
    uScene = u_scene(100, 100, 'uint8')
    mask = np.ones((10, 10), dtype='uint8')
    mask = mask[:, :]*256
    uScene.add_mask(0, 0, mask)
    uScene.plot_image()
