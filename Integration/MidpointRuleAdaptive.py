# -*- coding: utf-8 -*-
"""
Midpoints Rule with adaptive integration step size selection (h)

@author: ssklykov
"""
# %% Import Section
from sampleFunctions import SampleFuncIntegr as sfint
import inspect

# %% Algorithm implementation
def midpointRuleAdaptive(a:float,b:float,y,epsilon:float=0.01):
    """Midpoint Rule implementation. It demands [a,b] of interval for integration; y - function or method returning \
        single float number and accepting single float number; epsilon - difference of two subsequent calculated \
        integrals (condition for stopping) - absolute error"""
    if (a >= b):
        print("Incosistent interval assigning [a,b]")
        return None # returning null object instead of any result, even equal to zero
    elif not((inspect.isfunction(y)) or (inspect.ismethod(y))):
        print("Passed function y(x) isn't the defined (callable) method or function")
        return None
    else:
        h0 = 0.0; S0 = 0.0; S1 = 0.0; h1 = 0.0; nk = 1
        h0 = b - a  # Initial step size - very rough approximation
        kMax = 19 # Directly depends on maximum possible int number of operations
        k = 1 # counting number of divisions and iterations
        S0 = h0*y(a+(h0*0.5)) # 0th iteration
        # actual manual calculation 1th iteration for being sure that iterations will go properly
        h1 = h0/3; h12 = h1/2; nk *= 3
        S1 = h1*(y(a+h12) + y(a+3*h12) + y(a+5*h12))
        # print("0th iteration: ",S0)
        # print("1st iteration: ",S1)
        while ((k <= kMax) and (abs(S1-S0) > abs(S1)*epsilon)): # either there are too many step divisions, or the precision reached
            S0 = S1; h0 = h1 # for making an iteration step
            intSum = 0.0
            for i in range(1,nk): # actually, here n(k-1) used from the equation!
                intSum += y(a + (i-(5/6))*h1) + y(a + (i-(1/6))*h1) # h1 hasn't been updated!
            intSum *= h0
            S1 = (S0 + intSum)/3 # updated integral value
            nk *= 3 # stepping n(k) = 3*n(k-1)
            k += 1 # number of iterations
            h1 /= 3 # decreasing step size
            # print("at iteration",k,"integral value is",S1)
        return S1

# %% Testing values
nDigits = 4; a = 0; b = 2; nSample = 1; epsilon = 1e-3 # sensitive much more for epsilon!
fClass = sfint(nDigits,nSample) # making sample of the class contained the sample function
fClass2 = sfint(nDigits,nSample+2)
# %% Testing
integral = midpointRuleAdaptive(a,b,fClass.sampleF,epsilon)
integral = round(integral,3) # Actually, rounding strictly connects with epsilon!
print(integral," - calculated integral value")

integral2 = midpointRuleAdaptive(a,b,fClass2.sampleF,epsilon)
integral2 = round(integral2,nDigits) # Actually, rounding strictly connects with epsilon!
print(integral2," - calculated integral value")