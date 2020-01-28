"""
Sample function for checking ODE solvers
@author: ssklykov
"""
import math
"""
Please initialize this class with # of sample function (now available only 1) and precision of return value
"""
class SampleFunctions():
    def __init__(self,nSample:int=1,nDigits:int=3):
        self.nSample = nSample; self.nDigits = nDigits
    """
    Returning float value from calculated function
    """
    def getValue(self,x:float) -> float:
        if (self.nSample == 1):
            y = 3*math.pow(x,2)
            return round(y,self.nDigits)
