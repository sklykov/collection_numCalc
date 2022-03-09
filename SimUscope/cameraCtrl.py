# -*- coding: utf-8 -*-
"""
Camera wrapper for imports and further utilization in OOP manner.

@author: ssklykov
"""
# %% Imports


# %% Class wrapper
class PCOcamera():
    """Class for wrapping controls of the PCO camera and provided API features."""

    def __init__(self, exceptionsQueue):
        self.exceptionsQueue = exceptionsQueue  # For adding the exceptions that should stop the main program
        pass
