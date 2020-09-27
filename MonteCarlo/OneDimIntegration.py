# -*- coding: utf-8 -*-
"""
One dimensional integration using the Monte Carlo sampling method (Integral ~= V*<f> +- std)

@author: ssklykov
"""
# %% Import section + Adding a path to import module to sys.path
import os  # for getting cwd
# import platform  # for getting name of a running platform
import sys  # for setting another folder into a searching path for modules
import inspect  # for inspection that transferred to the developed method is a function like y = f(x)
import random
import math
# Done: the appending to sys.path should be OS agnostic - it should work both for Windows and Unix systems (the first one tested)
# Workaround to select proper separator for path handling (cleaned out)
current_path = os.getcwd()  # Inspect in a variable inspection window (instead of printing it out)
os.chdir("..")  # Unix-like command to go in a directory tree up
head_directory = os.getcwd()  # Checking that cwd changed
list_dirs = os.listdir(head_directory)  # Inspection only - get all directories in this repo
append_dir = os.path.join(head_directory, "Integration")  # Creating a path to other directory in this repo (should be OS agnostic)
# Checking the added folder and deleting extra ones
if (sys.path.count(append_dir) == 0):
    sys.path.append(append_dir)
elif (sys.path.count(append_dir) > 1):
    for i in (2, sys.path.count(append_dir)):
        sys.path.remove(append_dir)
sys_path = sys.path
from sampleFunctions import SampleFuncIntegr as yClass  # TODO: there are now many different names and it obfuscates the code!


# %% Monte Carlo one dimensional integral implementation
def MonteCarloInt1D(a: float, b: float, y, nSamples: int = 100):
    """ Implementation of 1D Monte Carlo integration of the input function y(x) using # Samples (nSamples)\
        in an interval [a,b]"""
    # Checking input parameters
    if (a >= b):
        print("Incosistent interval assigning [a,b]")
        interim = a; a = b; b = interim
    elif not((inspect.isfunction(y)) or (inspect.ismethod(y))):
        print("Passed function y(x) isn't the defined (callable) method or function")
        return None
    elif (nSamples <= 0):
        print("Please specify positive number of samples for generation")
        n = 100  # default value

    # Implementation itself
    sumF = 0.0; sumSquaredF = 0.0
    for i in range(nSamples):
        xRandom = random.random()  # no seed provided
        sumF += y((b-a)*xRandom + a)  # summation of f(x) values randomly distributed in the interval [a,b]
        sumSquaredF += y((b-a)*xRandom + a)*y((b-a)*xRandom + a)
    integralValue = 0.0; integralValue = ((b-a)/nSamples)*sumF
    meanFValue = sumF/nSamples; variance = math.sqrt((sumSquaredF/nSamples) - math.pow(meanFValue, 2))
    std = ((b-a)/math.sqrt(nSamples))*variance
    return(integralValue, std)

# %% Testing features
nDigits = 3; nFunction = 1; a = 0; b = 2; nSamples = 1000
y1 = yClass(nDigits, nFunction)
print("exact value of the integral is 1")
(integral, std) = MonteCarloInt1D(a, b, y1.sampleF, nSamples)
integral = round(integral, nDigits); std = round(std, nDigits)
print("integral value = ", integral, "with standard deviation", std, "calculated with", nSamples, "# of used x[i] points")
nSamples *= 10
(integral, std) = MonteCarloInt1D(a,b,y1.sampleF,nSamples)
integral = round(integral, nDigits); std = round(std,nDigits)
print("integral value = ", integral, "with standard deviation", std, "calculated with", nSamples, "# of used x[i] points")
nSamples *= 10
(integral, std) = MonteCarloInt1D(a,b,y1.sampleF,nSamples)
integral = round(integral,nDigits); std = round(std,nDigits)
print("integral value = ", integral, "with standard deviation", std, "calculated with", nSamples, "# of used x[i] points")
nSamples *= 5
(integral, std) = MonteCarloInt1D(a, b, y1.sampleF, nSamples)
integral = round(integral,nDigits); std = round(std,nDigits)
print("integral value = ", integral, "with standard deviation", std, "calculated with", nSamples, "# of used x[i] points")
