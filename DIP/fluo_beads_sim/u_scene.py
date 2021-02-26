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
    possible_img_types = ['ubyte', 'int', 'float']
    scene_image = np.zeros((width, height), dtype='ubyte')

    def __init__(self, width: int, height: int, image_type: str = 'ubyte'):
        width = abs(width)
        height = abs(height)
        if width > 0:
            self.width = width
        if height > 0:
            self.width = width
        if image_type in self.possible_img_types:
            self.image_type = image_type
        else:
            self.image_type = 'ubyte'
            print("Image type isn't recognized, initialized default 8bit gray image")
        if (width != 100) or (height != 100) or (image_type != 'ubyte'):
            self.scene_image = np.zeros((width, height), dtype=self.image_type)

    def plot_image(self):
        plt.figure()
        plt.imshow(self.scene_image, cmap=plt.cm.gray, aspect='auto', origin='lower',
                   extent=(0, self.height, 0, self.width))



# %% Testing class methods / construction
uScene = u_scene(150, 100, 'float')
uScene.plot_image()
