# -*- coding: utf-8 -*-
"""
Secant Method for finding real roots of equation f(x) = 0.

This method is really sensitive to the initial value, seems. E.g., the a from [a,b] interval doesn't work.
So, for the making this method more stable, the middle point of an interval [a,b] is chosen as the initial one

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


# %% Secant method implementation (NOT DIVERGED...)
def Secant(func, approxDerivative, a: float, b: float, epsilon: float, digitRound: int, showIteration: bool):
    # This method demand at least one initial root approximation
    x0 = (a+b)/2  # The worst case from the Newton's method implementation really doesn't work! NO Convergence
    i = 1  # Number of iteration steps
    x1 = x0 - func(x0)  # Next iteration step
    x2 = x1 - (func(x1)/approxDerivative(func, x1, x0))  # Final iteration step, utilized the previous two
    while (abs(func(x2)) > epsilon) and (i <= 5000):
        # The row below is for showing iteration process
        if showIteration:
            print("Iteration step #", i, "Root approximation is: ", round(x1, digitRound+1))
        i += 1; x0 = x1; x1 = x2; # Iteration step - update all previous calculated values
        x2 = x1 - func(x1)/approxDerivative(func, x1, x0)  # Final iteration step, utilized the previous two
    if (i > 5000):
        print("For some reason, there is too many ( > 5000) iteration steps made")
        return None
    return float(round(x1, digitRound))


# %% Testing of implemented methods
xRoot = Secant(f.example3, f.approxDerivative, a, b, epsilon, digitRound, showIteration)
print("The calculated real root is", xRoot)
