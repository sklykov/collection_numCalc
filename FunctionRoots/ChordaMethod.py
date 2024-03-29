# -*- coding: utf-8 -*-
"""
Chorda or "false position" method for finding real roots of equation f(x) = 0.

@author: sklykov
@license: The Unlicense
"""
# %% Import section
import AlgF  # Module contained a class with static methods - as examples of functions
from FuncPlotting import plotMyAwesomeFunc  # Plotting of function used for evaluation

# %% Parameters specification (instead of promting of an user for doing it)
a = -1; b = 4  # From testing: the convergance of this method can be really slow, try [-10,100] for a demo
nPoints = 500; epsilon = 0.01

# %% Plotting the function
f = AlgF.ExampleAlgF()  # Class construction
plotMyAwesomeFunc(f.example3, a, b, nPoints)  # plotting of 1st example of function
digitRound = 2; showIteration = True


# %% Chorda method implementation
def Chorda(func, a: float, b: float, epsilon: float, digitRound: int, showIteration: bool):
    """
    Calculates solution of equation f(x) = 0 by using chorda iterative approximations.

    Parameters
    ----------
    func : Callable
        Some function (method) returning by calling with input parameter x (float) some float number.
    a : float
        Lower border of an interval for root searching [a, b].
    b : float
        Higher border of an iterval for root searching.
    epsilon : float
        Relative precision of roots calculations for float numbers.
    digitRound : int
        Rounding of returning values to some relevant number after digit point.
    showIteration : bool
        For explicit showing number of iterations and root iteration.

    Returns
    -------
    float
        Single root of an equation like f(x) = 0.

    """
    # Epsilon - Presicion of equality the interesting function f(x) ~= 0
    # DigitRound - number of digits for rounding calculations (in some dependence to Epsilon)
    i = 0  # For calculation of a number of iteration steps
    if (func(a)*func(b) < 0):
        # The point there chorda is equal to zero (approximation to a real root)
        xChorda = (a*func(b)-b*func(a))/(func(b)-func(a))
        while (abs(func(xChorda)) > epsilon) and (i <= 5000):
            if (func(xChorda)*func(a) < 0):
                b = xChorda  # [a,b] => [a,xChorda] - approximation, visual interpretation - graphically
            elif (func(xChorda)*func(b) < 0):
                a = xChorda
            else:
                print("Apparantly, there is more than real 1 root or no roots...")
                return None
            # The row below is for showing iteration process
            if showIteration:
                print("Iteration step #", i, "Root approximation is: ", round(xChorda, digitRound+1))
            i += 1
            if (i > 5000):
                print("For some reason, there is too many ( > 5000) iteration steps made")
            xChorda = (a*func(b)-b*func(a))/(func(b)-func(a))  # For allowing iteration ad test next approximation to the root
        return float(round(xChorda, digitRound))  # In the end of while cycle
    else:
        print("There is no real roots between input a and b or more than 1 real root")
        return None


# %% Testing of implemented methods
xRoot = Chorda(f.example3, a, b, epsilon, digitRound, showIteration)
print("The calculated real root is", xRoot)
