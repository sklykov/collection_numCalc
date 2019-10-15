# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 2019
This script is saved for only keeping and possible reusing it in future.
This script is created for performing simple plotting using the matplotlib library
The main purpose is to collect commands for fast and fancy formatting of a produced graph
WARNING: the autocompletion doesn't work in Spyder IDE that is used for its creation
Generally, much information has been collected on the StackOverflow website
@author: ssklykov
"""
import matplotlib.pyplot as plt
import numpy as np

# preparing sample for plotting using np.arrays
stepX = 0.1; maxX = 5.0; minX = -5.0;
x = np.arange(minX, maxX+stepX, stepX)  # More handful than np.linspace (maybe)
x = np.around(x, 1)  # To guarantee values consistency
y = np.power(x,2)  # Calculation of x**2 for guaranting consistency of results (np.array - in and out)
y = np.around(y,2)

# Plotting and graph formatting - using OOP concepts
wSize = 1.2*(9.0/2.54); hSize = (9.0/2.54) # converting santimeters to inches
plt.rc('font',family='serif')  # I give up to find it without autocompletion in 'figure' or 'Axes' properties
fig = plt.figure(figsize=(wSize,hSize))  # fig itself... dpi only setting...
# axes = fig.add_subplot(1,1,1) # axes class (? - TAB doesn't found this property, Spyder IDE)
axes = plt.axes()  # alternative way for the line above for adding axes to the current figure
axes.plot(x,y,'r-',linewidth=3)  # The main plotting function
axes.set(xlim=(minX,maxX),ylim=(min(y),max(y)))  # set limits for axes
# set X axis settings
majorXTicks = np.arange(minX,maxX+0.1,1)
minorXTicks = np.arange(minX+0.5,maxX,0.5)
axes.set_xticks(majorXTicks)  # set the major ticks
axes.set_xticks(minorXTicks,minor=True)
axes.grid(which='both') # it's enought to specify it once
axes.set_xlabel('x',fontsize=12)
# set Y axis settings
majorYTicks = np.arange(min(y),max(y)+1.1,2)
# minorYTicks = np.arange(min(y)+1,max(y),1)
axes.set_yticks(majorYTicks)  # set the major ticks
# axes.set_yticks(minorYTicks,minor=True)
axes.set_ylabel('y',fontsize=12)
# axes.autoscale(enable=True)
fig.tight_layout()
# Figure saving
# fig.savefig("Parabola.png",dpi=300)
# fig.savefig("Parabola.tiff",dpi=200)
# fig.savefig("Parabola.eps",dpi=200)  # Somehow for presentation works bad!
# fig.savefig("Parabola.svg",dpi=200)  # For further using in Inkscape

# print(dir(axes)) # since autocompletion for "axes" doesn't work, this command print all attributes
# print(dir(fig))