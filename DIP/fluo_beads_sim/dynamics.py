# -*- coding: utf-8 -*-
"""
Simulation of diffusion / motion of microbeads.

@author: ssklykov
"""
# %% Imports


# %% Class definition
class diffusion():
    D = 1.0
    coordinates = []

    def __init__(self, D: float = 1):
        self.D = D

    def get_next_point(self, point):
        # Simulation for each coordinate in the approximated relation: r ~= sqrt(2)*x or r ~= sqrt(2)*y
        # The equation is taken from Qian, et al. (1991)
        pass
