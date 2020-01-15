# -*- coding: utf-8 -*-
"""
Simpson's Rule with automative (adaptive) selection of integration step by adjusting integral sum calculation for each
halving step (h[i] = h[i-1]/2). A user should only specify potentially the desired precision of integral sum calculation
Small modifications in comparison to TraperzoidalAutoAdaptive
@author: ssklykov
"""
# %% Import Section
from sampleFunctions import SampleFuncIntegr as sfint
import inspect

# %% Algorithm implementation
"""Simpson Integration implementation. It demands [a,b] of interval for integration; y - function or
method returning single float number and accepting single float number; epsilon - difference of two sub
sequent calculated integrals (condition for stopping) - absolute error"""
def SimpsonAutoAdaptive(a:float,b:float,y,epsilon:float=0.01):
    if (a >= b):
        print("Incosistent interval assigning [a,b]")
        return None # returning null object instead of any result, even equal to zero
    elif not((inspect.isfunction(y)) or (inspect.ismethod(y))):
        print("Passed function y(x) isn't the defined (callable) method or function")
        return None
    else:
        h0 = 0.0; T0 = 0.0; T1 = 0.0
        h0 = b - a  # Initial step size - very rough approximation
        T0 = (h0*(y(b)+y(a)))/2 # 0th iteration
        h1 = h0/2
        T1 = (T0/2) + h1*y(a+h1) # 1st iteration
        nMaxIterations = 30 # Maximum number of iterations, actually I don't use the integer number for calculation
        # with each iteration, number of new nodes involved in calculations growing to be as 2**j
        j = 1 # number of iterations
        S0 = T0; S1 = (4*T1 - T0)/3
        while ((abs(S1 - S0) > epsilon*S1) and (j < nMaxIterations)): # using relative error for exit this loop
            T0 = T1; h0 = h1; S0 = S1 # stepping further
            h1 = h0/2
            sumInt = 0.0; i = 0
            while ((a+h1*(2*i+1)) < b):
                sumInt += y(a+h1*(2*i+1)) # summing only new nodes for integral calculation
                i += 1
            # print(i,"number of nodes") # Debugging
            T1 = (T0/2) + h1*sumInt; S1 = (4*T1 - T0)/3; j += 1 # iteration step calculations
            # print(S1,"value at iteration",j) # Debugging
        print(j,"- overall number of iterations (halfing)")
        return S1

# %% Parameters for testing
nDigits = 8; a = 0; b = 2; nSample = 1; epsilon = 1e-8
fClass = sfint(nDigits,nSample) # making sample of the class contained the sample function
fClass2 = sfint(nDigits,nSample+2)
# %% Testing
integral = SimpsonAutoAdaptive(a,b,fClass.sampleF,epsilon)
# integral = round(integral,3) # Actually, rounding strictly connects with epsilon!
print(integral," - calculated integral value")

integral2 = SimpsonAutoAdaptive(a,b,fClass2.sampleF,epsilon)
integral2 = round(integral2,nDigits) # Actually, rounding strictly connects with epsilon!
print(integral2," - calculated integral value")