#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper classes (?) with some repeatable functionality for U8 (...) images

@author: ssklykov
"""


class WrapUbyteImg():
    """
    Wrapper class for U8 image (for calculating some attributes all at once)
    """
    # %% Imports (denepndents)
    import numpy as np
    # %% Some attributes of the class pre-initialization
    maxPixelValue = np.uint8(0)
    minPixelValue = np.uint8(0)
    x_size = 0
    y_size = 0
    image = np.zeros((2, 2), dtype='uint8')

    def __init__(self, uByte_image):
        """
        Constructing the sample of a class with some preloaded sample (maybe will be exchanged)
        Calculating of image sizes, max and min pixel values in it
        """
        import numpy as np  # Also required locally
        from skimage.util import img_as_ubyte
        object_identifier = str(type(uByte_image))  # Checking the type of passing to the constructor object
        # Below - kind of type checking
        if "ndarray" in object_identifier:
            self.image = img_as_ubyte(uByte_image, force_copy=True)
        else:
            try:
                if "list" in object_identifier:
                    rows = len(uByte_image)
                    cols = len(uByte_image[0])
                    if (rows > 0) and (cols > 0):
                        self.image = np.uint8(np.array(uByte_image))
                    else:
                        raise("ZeroInputList")
                else:
                    raise("NotProperInputImage")
            except AttributeError:
                raise("NotProperInputImage")
        (self.y_size, self.x_size) = self.image.shape
        self.maxPixelValue = np.max(self.image)
        self.minPixelValue = np.min(self.image)
