# -*- coding: utf-8 -*-
"""
Bisection method of finding real roots (f(x)=0) of a function f(x).

Two main restrictions:
    1) Somehow the interval for searching of roots should be specified ([a,b])4
    2) The function of interest should be somehow specified
TODO for future: GUI specification of input parameters for algebraic function or even function itself.

@author: sklykov
@license: The Unlicense
"""
# %% Import section
from AlgebraicFunctionEx import fVal as f
from FuncPlotting import plotMyExampleFunc
# %% Primitive interaction with user - asking for input lower and higher borders of a searching interval
# The IPython console is used for reading and typing of input values...
# For testing the feature of the primitive prompting of an user for input values, this module should be imported
if __name__ != "__main__":
    try:
        a = float(input("Enter the lower border of a searching interval: "))
    except ValueError:
        print("Wrong input value - the lower border")
    try:
        b = float(input("Enter the higher border of a searching interval: "))
    except ValueError:
        print("Wrong input value - the lower border")
else:
    # Testing values along with test function - optimizied for example below
    a = -1
    b = 4


# %% Plotting of function
nValues = 500  # For plotting the function under revision
nFunc = 3  # Number of an function from examples
plotMyExampleFunc(f, nFunc, a, b, nValues)  # calling related function for plotting


# %% Bisection method implementation
def Bisection(func, nFunc: int, a: float, b: float, epsilon: float, digitRound: int, showIteration: bool):
    # Epsilon - Presicion of equality the interesting function f(x) ~= 0
    # DigitRound - number of digits for rounding calculations (in some dependence to Epsilon)
    # nFunc - number of a predefined function
    i = 0  # For calculation of a number of iteration steps, the upper bound for a number of iteration (divergance)
    if (func(a, nFunc)*func(b, nFunc) < 0):
        xMiddle = (a+b)/2  # The middle point in interval [a,b]
        while (abs(func(xMiddle, nFunc)) > epsilon) and (i <= 5000):
            if (func(xMiddle, nFunc)*func(a, nFunc) < 0):
                b = xMiddle  # [a,b] => [a,xMiddle] - the bisection operation
            elif (func(xMiddle, nFunc)*func(b, nFunc) < 0):
                a = xMiddle
            else:
                print("Apparantly, there is more than real 1 root or no roots...")
                return None
            # The row below is for showing iteration process
            if showIteration:
                print("Iteration step #", i, "Root approximation is: ", round(xMiddle, digitRound+1))
            i += 1
            if (i > 5000):
                print("For some reason, there is too many (> 5000) iteration steps made")
            xMiddle = (a+b)/2  # For allowing iteration ad test next approximation to the root
        return float(round(xMiddle, digitRound))  # In the end of while cycle
    else:
        print("There is no real roots between input a and b or more than 1 real root")
        return None


# %% Testing of implemented methods
xRoot = Bisection(f, nFunc, a, b, 0.01, 2, True)
print("The calculated real root is", xRoot)
