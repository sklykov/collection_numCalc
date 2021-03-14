# -*- coding: utf-8 -*-
"""
Container for building a scene with fluorescent objects

@author: ssklykov
"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte


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
        # print(nRows, nCols, " - sizes of input mask")  # Checking
        if (nRows == 0) or (nCols == 0):
            raise(IndexError('Provided mask is empty along some of its axis'))
        # TODO
        # 1) x_start, y_start - could be negative, make the logic for it
        # 2) make checking of input mask parameters (more than 255 sum value for uint8 image and mask for example)
        # 3) add logic to below
        if ((x_start + nRows) < self.height):  # checking that fast sum over x axis could be performed
            if ((y_start + nCols) < self.width):  # checking that starting and ending of summing not of bound
                for j in range(y_start, y_start + nCols):
                    if np.max(self.scene_image[x_start:x_start+nRows, j] + mask[:, j-y_start]) <= self.maxPixelValue:
                        self.scene_image[x_start:x_start+nRows, j] += mask[:, j-y_start]
                    else:
                        for i in range(x_start, x_start+nRows):
                            pixels_sum = self.scene_image[i, j] + mask[i-x_start, j-y_start]
                            if (pixels_sum) <= self.maxPixelValue:
                                if self.image_type == 'uint8':
                                    self.scene_image[i, j] = np.uint8(pixels_sum)
                            else:
                                self.scene_image[i, j] = self.maxPixelValue
            else:
                for j in range(y_start, self.width):
                    self.scene_image[x_start:x_start+nRows, j] += mask[:, j-y_start]

    def plot_image(self):
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
    uScene.add_mask(1, 1, mask)
    uScene.plot_image()
