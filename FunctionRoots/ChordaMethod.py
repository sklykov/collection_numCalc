# -*- coding: utf-8 -*-
"""
Chorda or "false position" method for finding real roots of equation f(x) = 0
Developed in the Spyder IDE
@author: ssklykov
"""
# %% Import section
import AlgF  # Module contained a class with static methods - as examples of functions
from FuncPlotting import plotMyAwesomeFunc # Plotting

# %% Parameters specification (instead of promting of an user for doing it)
a = -1; b = 4  # From testing: the convergance of this method can be really slow, try [-10,100] for a demo
nPoints = 500; epsilon = 0.01

# %% Plotting the function
f = AlgF.ExampleAlgF() # Class construction
plotMyAwesomeFunc(f.example3,a,b,nPoints) # plotting of 1st example of function
digitRound = 2; showIteration = True

# %% Chorda method implementation
def Chorda(func,a:float,b:float,epsilon:float,digitRound:int,showIteration:bool):
    # Epsilon - Presicion of equality the interesting function f(x) ~= 0
    # DigitRound - number of digits for rounding calculations (in some dependence to Epsilon)
    i = 0  # For calculation of a number of iteration steps
    if (func(a)*func(b) < 0):
        xChorda = (a*func(b)-b*func(a))/(func(b)-func(a)) # The point there chorda is equal to zero (approximation to a real root)
        while (abs(func(xChorda)) > epsilon) and (i <= 5000):
            if (func(xChorda)*func(a) < 0):
               b = xChorda # [a,b] => [a,xChorda] - approximation, visual interpretation - graphically
            elif (func(xChorda)*func(b) < 0):
                a = xChorda
            else:
                print("Apparantly, there is more than real 1 root or no roots...")
                return None
            # The row below is for showing iteration process
            if showIteration: print("Iteration step #",i,"Root approximation is: ",round(xChorda,digitRound+1))
            i += 1
            if (i > 5000): print("For some reason, there is too many ( > 5000) iteration steps made")
            xChorda = (a*func(b)-b*func(a))/(func(b)-func(a))  # For allowing iteration ad test next approximation to the root
        return float(round(xChorda,digitRound)) # In the end of while cycle
    else:
        print("There is no real roots between input a and b or more than 1 real root")
        return None

# %% Testing of implemented methods
xRoot = Chorda(f.example3,a,b,epsilon,digitRound,showIteration)
print("The calculated real root is",xRoot)