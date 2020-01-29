# -*- coding: utf-8 -*-
"""
One dimensional integration using the Monte Carlo sampling method (Integral ~= V*<f> +- std)

@author: ssklykov
"""

# %% Import section
import os # for getting cwd
import platform # for getting name of a running platform
import sys # for setting another folder into a searching path for modules
import inspect # for inspection that transferred to the developed method is a function like y = f(x)
import random
import math
# TODO: Take care and test path representations for Linux system as well (now it's written on Win)
# print(platform.system())
# Workaround to select proper separator for path handling
if (platform.system() == "Windows"):
        charPathSeparator = "\\"
        print("This script running on Windows system")
elif(platform.system() == "Linux"):
    charPathSeparator = "/"
# Variables for path handling
currentPath = os.getcwd()
indexOfLastSlash = 0; i = 0
# Deletion of path to current subfolder (kind of command "cd..")
while(i < len(currentPath)):
    if (currentPath[i] == charPathSeparator):
        indexOfLastSlash = i
    i += 1
currentPath2 = currentPath[0:indexOfLastSlash] # Get the path with command "cd.." implemented to the working directory
currentPath2 += "\\Integration" # going to the another folder
# print(currentPath2)
# Adding directory to the PATH variable for importing (only as a workaround)
sys.path.append(currentPath2)
from sampleFunctions import SampleFuncIntegr as yClass # TODO: there are now many different names and it obfuscates the code!

# %% Monte Carlo one dimensional integral implementation
def MonteCarloInt1D(a:float,b:float,y,nSamples:int=100):
    """ Implementation of 1D Monte Carlo integration of the input function y(x) using # Samples (nSamples)\
        in an interval [a,b]"""
    # Checking input parameters
    if (a >= b):
        print("Incosistent interval assigning [a,b]")
        return None # returning null object instead of any result, even equal to zero
    elif not((inspect.isfunction(y)) or (inspect.ismethod(y))):
        print("Passed function y(x) isn't the defined (callable) method or function")
        return None
    elif (nSamples <= 0):
        print("Please specify positive number of samples for generation")

    # Implementation itself
    sumF = 0.0; sumSquaredF = 0.0
    for i in range(nSamples):
        xRandom = random.random() # no seed provided
        sumF += y((b-a)*xRandom + a) # summation of f(x) values randomly distributed in the interval [a,b]
        sumSquaredF += y((b-a)*xRandom + a)*y((b-a)*xRandom + a)
    integralValue = 0.0; integralValue = ((b-a)/nSamples)*sumF
    meanFValue = sumF/nSamples; variance = math.sqrt((sumSquaredF/nSamples) - math.pow(meanFValue,2))
    std = ((b-a)/math.sqrt(nSamples))*variance
    return(integralValue,std)

# %% Testing features
nDigits = 3; nFunction = 1; a = 0; b = 2; nSamples = 1000
y1 = yClass(nDigits,nFunction)
print("exact value of the integral is 1")
(integral,std) = MonteCarloInt1D(a,b,y1.sampleF,nSamples)
integral = round(integral,nDigits); std = round(std,nDigits)
print("integral value = ",integral,"with standard deviation",std,"calculated with",nSamples,"# of used x[i] points")
nSamples *= 10
(integral,std) = MonteCarloInt1D(a,b,y1.sampleF,nSamples)
integral = round(integral,nDigits); std = round(std,nDigits)
print("integral value = ",integral,"with standard deviation",std,"calculated with",nSamples,"# of used x[i] points")
nSamples *= 10
(integral,std) = MonteCarloInt1D(a,b,y1.sampleF,nSamples)
integral = round(integral,nDigits); std = round(std,nDigits)
print("integral value = ",integral,"with standard deviation",std,"calculated with",nSamples,"# of used x[i] points")
nSamples *= 5
(integral,std) = MonteCarloInt1D(a,b,y1.sampleF,nSamples)
integral = round(integral,nDigits); std = round(std,nDigits)
print("integral value = ",integral,"with standard deviation",std,"calculated with",nSamples,"# of used x[i] points")