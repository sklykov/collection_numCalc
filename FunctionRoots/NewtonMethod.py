# -*- coding: utf-8 -*-
"""
Newton's method for finding real roots of equation f(x) = 0.

@author: sklykov
@license: The Unlicense
"""
# %% Import section
import AlgF  # Module contained a class with static methods - as examples of functions
from FuncPlotting import plotMyAwesomeFunc  # Plotting

# %% Parameters specification (instead of promting of an user for doing it)
a = -1; b = 4  # From testing: the convergance of this method can be really slow, try [-10,100] for a demo
nPoints = 500; epsilon = 0.01

# %% Plotting the function
f = AlgF.ExampleAlgF()  # Class construction
plotMyAwesomeFunc(f.example3, a, b, nPoints)  # plotting of 1st example of function
digitRound = 2; showIteration = True


# %% Newton method implementation
def NewtonMethod(func, funcDeriv, a: float, b: float, epsilon: float, digitRound: int, showIteration: bool):
    x0 = a  # Initial point - the starting of interval (for testing case - the worst approximation, from the graph plotted)
    i = 1  # Number of iteration steps
    x1 = x0 - (func(x0)/funcDeriv(x0, epsilon))
    while (abs(func(x1)) > epsilon) and (i <= 5000):
        # The row below is for showing iteration process
        if showIteration:
            print("Iteration step #", i, "Root approximation is: ", round(x1, digitRound+1))
        i += 1; x0 = x1  # For making another step at the line below
        x1 = x0 - (func(x0)/funcDeriv(x0, epsilon))
    if (i > 5000):
        print("For some reason, there is too many ( > 5000) iteration steps made")
        return None
    return float(round(x1, digitRound))


# %% Testing of implemented methods
xRoot = NewtonMethod(f.example3, f.example3Derivative, a, b, epsilon, digitRound, showIteration)
print("The calculated real root is", xRoot)
