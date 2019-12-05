# -*- coding: utf-8 -*-
"""
Bisection method of finding real roots (f(x)=0) of a function f(x)
Two main restrictions:
    1) Somehow the interval for searching of roots should be specified ([a,b])4
    2) The function of interest should be somehow specified
TODO for future: GUI specification of input parameters for algebraic function or even function itself
Developed in the Spyder IDE
@author: ssklykov
"""
# %% Import section
import numpy as np
from AlgebraicFunctionEx import fVal as f
from matplotlib import pyplot as plt

# %% Primitive interaction with user - asking for input lower and higher borders of a searching interval
# The IPython console is used for reading and typing of input values...
# For testing the feature of the primitive prompting of an user for input values, this module should be imported
if __name__ == "main":
    try:
        a = float(input("Enter the lower border of a searching interval: "))
    except ValueError:
        print("Wrong input value - the lower border")
    try:
        b = float(input("Enter the higher border of a searching interval: "))
    except ValueError:
        print("Wrong input value - the lower border")
else:
    a = -2; b = 3  # Testing values along with test function


# %% Plotting of function
nValues = 500  # For plotting the function under revision
x = np.linspace(a,b,nValues)  # Generation of evenly spaced X values
y = np.zeros(nValues,dtype=float) # Init Y values
x0 = np.zeros(nValues,dtype=float) # For marking the zero X axis
for i in range(nValues):
    y[i] = f(x[i],1)
# Setting up the graphic properties. More details - in relevant section of this repository
plt.rc('font',family='serif') # Trying to use open-source font
fig = plt.figure()
axes = plt.axes()
axes.plot(x,y,'r-',linewidth=3)  # The main plotting function - plotting of algebraic function of interest
axes.plot(x,x0,'b-',linewidth=3)  # Plotting the X axis
axes.set(xlim=(min(x),max(x)),ylim=(min(y),max(y)))  # set limits for axes
plt.grid()

# %% Bisection method implementation
def Bisection(func,a:float,b:float,epsilon:float,digitRound:int,showIteration:bool):
    # Epsilon - Presicion of equality the interesting function f(x) ~= 0
    # DigitRound - rounding calculations (in some dependence to Epsilon)
    i = 0  # For calculation of a number of iteration steps
    if (func(a)*func(b)<0):
        xMiddle = (a+b)/2  # The middle point in interval [a,b]
        while (abs(func(xMiddle)) > epsilon):
            if (func(xMiddle)*func(a) < 0):
               b = xMiddle # [a,b] => [a,xMiddle] - the bisection operation
            else:
                a = xMiddle
            # The row below is for showing iteration process
            if showIteration: i += 1; print("Iteration step #",i,"Root approximation is: ",round(xMiddle,digitRound+1))
            xMiddle = (a+b)/2  # For allowing iteration ad test next approximation to the root
        return float(round(xMiddle,digitRound)) # In the end of while cycle
    else:
        print("There is no real roots between input a and b")
        return None


# %% Testing of implemented methods
xRoot = Bisection(f,a,b,0.05,2,True)
print("The calculated root is",xRoot)
