# -*- coding: utf-8 -*-
"""
Sample (real defined, single argument) functions (y= f(x)) for testing numerical integration concepts.

@author: sklykov
@license: The Unlicense
"""
# %% Import section
import math


# %% Class definition - with sample functions realised as non-static methods
class SampleFuncIntegr():
    """Class for getting some sample functions for testing integration algorithms."""

    nOfSample: int = 1
    nDigits: int = 3

    def __init__(self, nDigits: int = 3, nOfSample: int = 1):
        """
        Specify the precision of returning float value (nDigits) and number of sample functions.

        Parameters
        ----------
        nDigits : int, optional
            Number of returning digits after float point. The default is 3.
        nOfSample : int, optional
            Selector for the sample function. The default is 1.
            if nOfSample == 1: return x^3 - x^2 - 1.5*x
            if nOfSample == 2: return x^2*exp(-x^2)
            if nOfSample == 3: return x^3*exp(-x^2)
            else: return sinh(x)

        Returns
        -------
        None.

        """
        self.nDigits = nDigits
        self.nOfSample = nOfSample

    def sampleF(self, x: float):
        """
        Return function value depending on the used initialization parameter nOfSample.

        Parameters
        ----------
        x : float
            Variable for the function.

        Returns
        -------
        float
            Function value.

        """
        y = 0
        if (self.nOfSample == 1):
            y = 3.0*math.pow(x, 2) - 2.0*x - 1.5  # F(x) = x^3 - x^2 - 1.5*x - analytical integral
        elif (self.nOfSample == 2):
            x2 = math.pow(x, 2)
            y = x2*math.exp(-x2)  # difficult for analytical integration
        elif (self.nOfSample == 3):
            y = math.pow(x, 3)*math.exp(-math.pow(x, 2))
        else:
            y = math.sinh(x)

        return round(y, self.nDigits)
