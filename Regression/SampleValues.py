# -*- coding: utf-8 -*-
"""
Class for modelling samples values (measurments with some deviations)
Developed in Spyder IDE
@author: ssklykov
"""
# %%  "Dependecies" - imports
import numpy as np
import math
# %% Class itself
"""Class for generating sample values"""
class GenerateSample():
    xMin = 0; xMax = 1; nSamples = 5; percentError = 10; a = 1; b = 0; nDigits = 3
    def __init__(self,a:float,b:float,xMin:float,xMax:float,nSamples:int,percentError:int,nDigits:int):
        self.xMin = xMin; self.xMax = xMax; self.a = a; self.b = b; self.nDigits = nDigits
        self.nSamples = nSamples; self.percentError = percentError

    """ Mimicring the measurements with errors """
    def generateSamplePoint(self,x:float):
        n = 10  # modelling how many points "have been measured" for calcution mean values and standard deviations
        y = np.zeros(n,dtype=float); sum = 0.0
        for i in range(n):
            # modelling  a measurment with an error
            rand1 = 1+0.01*(np.random.randint(-self.percentError,self.percentError+1)) # generation of rand coeff 1
            rand2 = 1+0.01*(np.random.randint(-self.percentError,self.percentError+1)) # rand coeff 2
            y[i] = self.a*rand1*x + self.b*rand2
            sum += y[i]
        # Calculation of returning values
        yMean = sum / n # Mean value
        sum = 0.0 # making again it zero!
        for i in range(n):
            sum += pow((y[i]-yMean),2) # Calculation of (y[i] - yMean)**2
        yStD = math.sqrt(sum/(n-1))  # Calculation of estimation of a standard deviation
        return (yMean,yStD)

    """ Making the samples """
    def generateSampleValues(self):
        x = np.linspace(self.xMin,self.xMax,self.nSamples)
        for i in range(self.nSamples):
            x[i] = round(x[i],self.nDigits)

        yMean = np.zeros(self.nSamples,dtype=float)
        yStD = np.zeros(self.nSamples,dtype=float)
        # Generation sample values
        for i in range(self.nSamples):
            (yMean[i],yStD[i]) = self.generateSamplePoint(x[i])
            yMean[i] = round(yMean[i],self.nDigits)
            yStD[i] = round(yStD[i],self.nDigits)
        return (x,yMean,yStD)