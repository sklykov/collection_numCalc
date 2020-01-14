# -*- coding: utf-8 -*-
"""
Sample (real defined, single argument) functions (y= f(x)) for testing numerical integration concepts
For class initialization, please, specify the precision of returning value and adress to the sample functions
by calling method "sampleF" with number_of_sample function
@author: ssklykov
"""
# %% Import section
import math

# Class definition - with sample functions realised as non-static methods
class SampleFuncIntegr():
    nOfSample = 1; nDigits = 3
    def __init__(self,nDigits:int=3,nOfSample:int=1):
        self.nDigits = nDigits; self.nOfSample = nOfSample

    def sampleF(self,x:float):
        y = 0
        if (self.nOfSample == 1):
            y = 3*math.pow(x,2) - 2*x - 1.5 # F(x) = x^3 - x^2 - 1.5*x - analytical integral
        elif(self.nOfSample == 2):
            x2 = math.pow(x,2)
            y = x2*math.exp(-x2)  # difficult for analytical integration

        return round(y,self.nDigits)

