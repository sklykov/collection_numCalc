# -*- coding: utf-8 -*-
"""
Trapezoidal Rule with adaptive selection of integration step.

It is achieved by adjusting integral sum calculation for each halving step (h[i] = h[i-1]/2).
A user should only specify the desired precision of integral sum calculation.

@author: sklykov
@license: The Unlicense
"""
# %% Import Section
from sampleFunctions import SampleFuncIntegr as sfint
import inspect


# %% Algorithm implementation
def TrapezoidalAutoAdaptive(a: float, b: float, y, epsilon: float = 0.01) -> float:
    """
    Trapezoidal Integration implementation.

    It demands [a,b] of interval for integration; y - function or method returning single float number
    and accepting single float number; epsilon - difference of two sub sequent calculated integrals
    (condition for stopping) - absolute error.

    Parameters
    ----------
    a : float
        Lower bond of integration interval.
    b : float
        Higher bond of integration interval.
    y : callable function / method
        Function for which integration undergoes.
    epsilon : float, optional
        Difference between integration sum that is considered to be small enough. The default is 0.01.

    Returns
    -------
    float
        Integral sum.

    """
    if (a >= b):
        print("Incosistent interval assigning [a,b]")
        return None  # returning null object instead of any result, even equal to zero
    elif not ((inspect.isfunction(y)) or (inspect.ismethod(y))):
        print("Passed function y(x) isn't the defined (callable) method or function")
        return None
    else:
        h0 = 0.0; T0 = 0.0; T1 = 0.0
        h0 = b - a  # Initial step size - very rough approximation
        T0 = (h0*(y(b)+y(a)))/2  # 0th iteration
        h1 = h0/2
        T1 = (T0/2) + h1*y(a+h1)  # 1st iteration
        nMaxIterations = 30  # Maximum number of iterations, actually I don't use the integer number for calculation
        # with each iteration, number of new nodes involved in calculations growing to be as 2**j
        j = 1  # number of iterations
        while ((abs(T1 - T0) > epsilon*T1) and (j < nMaxIterations)):  # using relative error for exit this loop
            T0 = T1; h0 = h1  # stepping further
            h1 = h0/2
            sumInt = 0.0; i = 0
            while ((a+h1*(2*i+1)) < b):
                sumInt += y(a+h1*(2*i+1))  # summing only new nodes for integral calculation
                i += 1
            # print(i,"number of nodes") # Debugging
            T1 = (T0/2) + h1*sumInt; j += 1
            # print(T1,"value at iteration",j) # Debugging
        print(j, "- overall number of iterations (halfing)")
        return T1


# %% Parameters for testing
nDigits = 8; a = 0; b = 2; nSample = 1; epsilon = 1e-8
fClass = sfint(nDigits, nSample)  # making sample of the class contained the sample function0
fClass2 = sfint(nDigits, nSample+2)

# %% Testing
integral = TrapezoidalAutoAdaptive(a, b, fClass. sampleF, epsilon)
integral = round(integral, nDigits)  # Actually, rounding strictly connects with epsilon!
print(integral, " - calculated integral value")

integral2 = TrapezoidalAutoAdaptive(a, b, fClass2.sampleF, epsilon)
integral2 = round(integral2, nDigits)  # Actually, rounding strictly connects with epsilon!
print(integral2, " - calculated integral value")
