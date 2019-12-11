# -*- coding: utf-8 -*-
"""
Modelling of function values for testing of interpolation
Developed in the Spyder IDE
@author: ssklykov
"""
import numpy as np
import math

class InterpolSamples():
    n = 10 # Default number of sampling points for performing interpolation
    percentError = 5  # Default maximum percentage for deviations of function values from tabulated ones
    lowestVal = 0; highestVal = 5;
    def __init__(self,n:int,percentError:int,lowestVal:float,highestVal:float):
        if (n > 1):
            self.n = n
        else:
            print("The specified number of points less than 1")
            self.n = 10  # Default value remains
        if (percentError >= 0) and (percentError <= 100):
            self.percentError = percentError
        else:
            print("The Specified number of points less than 1")
            self.percentError = 5 # Default value remains
        self.lowestVal = lowestVal; self.highestVal = highestVal  # [a,b] interval

    """
    The main function for sample values generation
    return x values (evenly spaced values in [a,b] range), yPure - tabulated values of y(x), yErrors - yPure + errors
    """
    def valuesForInterpol(self):
        yError = np.zeros(self.n) # Initialization of returning values with errors
        yPure = np.zeros(self.n) # Initialization of returning values without errors
        x = np.linspace(self.lowestVal,self.highestVal,self.n)  # Number of points + 1 because of including starting point
        for i in range(self.n):
            yPure[i] = InterpolSamples.sampleFunction(x[i])
            yError[i] = round(InterpolSamples.sampleFunction(x[i])*(1 + 0.01*
                  (np.random.randint(-self.percentError,self.percentError+1))),3)
        return (x,yPure,yError)

    """
    Sample function (x**2)*exp(-0.5*x)
    """
    @staticmethod
    def sampleFunction(x:float):
        return round(pow(x,3)*math.exp(-x),3)

    # Only for testing the class creation
    def printValues(self):
        print(self.n,self.percentError,self.lowestVal,self.highestVal,"- values associated with class")

