# -*- coding: utf-8 -*-
"""
Experiments with simulation of images of fluorescent beads

@author: ssklykov
"""
# %% General imports
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from u_scene import u_scene

# %% class definition
class image_beads():

    # default values
    width = 10
    height = 10
    possible_img_types = ['uint8', 'uint16', 'float']
    character_size = 5
    image_type = 'uint8'
    bead_types = ["even round", "gaussian round", "uneven round"]
    bead_img = np.zeros((height, width), dtype= image_type)

    def __init__(self, image_type: str = 'uint8', character_size: int = 5, bead_type: str = "even round"):
        if image_type in self.possible_img_types:
            self.image_type = image_type
        else:
            self.image_type = 'uint8'
            print("Image type isn't recognized, initialized default 8bit gray image")
        if bead_type in self.bead_types:
            self.bead_type = image_type
        else:
            self.image_type = 'uint8'
            print("Image type isn't recognized, initialized default 8bit gray image")
        if (character_size != 5) or (image_type != 'uint8'):
            width = character_size*2
            height = character_size*2
            self.scene_image = np.zeros((height, width), dtype=self.image_type)
            if image_type == 'uint16':
                self.maxPixelValue = 65535
            else:
                self.maxPixelValue = 1.0  # According to the specification of scikit-image


# %% General parameters
width = 1000
height = 1000

# %% Testing features
if __name__ == '__main__':
    pic = u_scene(width, height, 'uint8')
