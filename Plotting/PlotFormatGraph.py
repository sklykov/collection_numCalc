# -*- coding: utf-8 -*-
"""
This script is saved for only keeping and possible reusing it in future.
This script has been created for performing simple plotting using the matplotlib library
The main purpose is to collect commands for fast and fancy formatting of a produced graph
WARNING: the autocompletion doesn't work in Spyder IDE that has been used for its creation
Generally, the most useful information about the used above methods has been collected
on the StackOverflow website, instead of using simple way of autocompletion / DocStrings reading
@author: ssklykov
"""
import matplotlib.pyplot as plt
import numpy as np

# preparing sample for plotting using np.arrays
stepX = 0.1; maxX = 5.0; minX = -5.0;
x = np.arange(minX, maxX+stepX, stepX)  # More handful than np.linspace (maybe)
x = np.around(x, 2)  # To guarantee values consistency (only 2 significant digits in float numbers)
y = np.power(x,2)  # Calculation of x**2 for guaranting consistency of results (np.array - in and out)
y = np.around(y,1) # To guarantee values consistency (only 2 significant digits in float numbers)

# Plotting and graph formatting - using OOP concepts
wSize = 1.2*(9.0/2.54); hSize = (9.0/2.54) # converting santimeters to inches
plt.rc('font',family='serif')  # I give up to find it without autocompletion in 'figure' or 'Axes' properties
fig = plt.figure(figsize=(wSize,hSize))  # fig itself... dpi only setting...
# axes = fig.add_subplot(1,1,1) # axes class
axes = plt.axes()  # the alternative way, in comparison to the line above, for adding axes to the current figure window
axes.plot(x,y,'r-',linewidth=3)  # The main plotting function
axes.set(xlim=(minX,maxX),ylim=(min(y),max(y)))  # set limits for axes

# set X axis settings (I gave up to find "OOP" way of setting properties in the Spyder IDE)
majorXTicks = np.arange(minX,maxX+0.1,1)
minorXTicks = np.arange(minX+0.5,maxX,0.5)
axes.set_xticks(majorXTicks)  # set the major ticks
axes.set_xticks(minorXTicks,minor=True)
axes.set_xlabel('x',fontsize=14,fontfamily='Liberation Serif')

# set Y axis settings
majorYTicks = np.arange(min(y),max(y)+1.1,2)
axes.set_yticks(majorYTicks)  # set the major ticks
# axes.set_yticks(minorYTicks,minor=True)
axes.set_ylabel('y',fontsize=14,fontfamily='Liberation Serif')

# Final preparations
axes.grid(which='both') # it's enought to specify it once
fig.tight_layout()  # making filling of figure box tight

# Figure saving
# fig.savefig("Parabola.png",dpi=300)
# fig.savefig("Parabola.tiff",dpi=200)
# fig.savefig("Parabola.eps",dpi=200)  # Somehow for presentation works bad!
# fig.savefig("Parabola.svg",dpi=200)  # For further using in the Inkscape

#print(dir(axes)) # since autocompletion for "axes" doesn't work, this command print all attributes
# print(dir(fig))