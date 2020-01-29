# -*- coding: utf-8 -*-
"""
Sample function for checking ODE solvers
@author: ssklykov
"""
import math

class SampleFunctions():
    """Please initialize this class with # of sample function (now available only 1) and precision of return value"""
    def __init__(self,nSample:int=1,nDigits:int=3):
        self.nSample = nSample; self.nDigits = nDigits

    def getValue(self,x:float) -> float:
        """Returning a float value from a predefined function"""
        if (self.nSample == 1):
            y = math.pow(x,2)
            return round(y,self.nDigits)

    def getTheoreticalValue(self,x:float) -> float:
        """Returning values of theoretical (solution) values"""
        if (self.nSample == 1):
            y = math.pow(x,3)/3
            return round(y,self.nDigits)
