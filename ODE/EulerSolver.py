# -*- coding: utf-8 -*-
"""
Implementation of simple Euler solver and Euler solver with correction for 1D simple ODE like: y'(x) = f(x)

@author: ssklykov
"""

#%% Import section
from SampleF import SampleFunctions
import os

#%% Simple Euler solver
def eulerSimple(f,h:float,y0:float,xFinish:float):
    pass

#%% Testing
y1 = SampleFunctions(1,4)
print(os.getcwd())
print(y1.getValue(2))