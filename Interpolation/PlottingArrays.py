# -*- coding: utf-8 -*-
"""
Plotting of input arrays - call different functions to plot different sets of input arrays
@author: ssklykov
"""
# %% Import section
from matplotlib import pyplot as plt

# %% Plotting of two arrays with identical [a,b] in X axis
"""Plotting of two input arrays (y1(x),y2(x)) sharing same borders for X axis"""
def plotTwoArrSameX(x,y1,y2,title:str="Samples",label1:str="y1",label2:str="y2"):
    plt.rc('font',family='serif') # Trying to use open-source font
    plt.figure(); axes = plt.axes(); plt.title(title)
    if (min(y1) > min(y2)):
        minY = min(y2)
    else:
        minY = min(y1)
    if (max(y1) < max(y2)):
        maxY = max(y2)
    else:
        maxY = max(y1)
    axes.set(xlim=(min(x),max(x)),ylim=(minY,maxY*1.05))  # set limits for axes
    axes.plot(x,y1,'ro-',linewidth=2,label=label1)
    axes.plot(x,y2,'bo-',linewidth=2,label=label2)
    axes.legend() # Necessary for displaying the legend
    plt.grid()

# %% Plotting of single array
"""Plotting of single array function y(x) = x """
def plotSingleArrSameX(x,y,title:str="Sample"):
    plt.rc('font',family='serif') # Trying to use open-source font
    plt.figure(); axes = plt.axes(); plt.title(title)
    axes.set(xlim=(min(x),max(x)),ylim=(min(y),max(y)*1.05))  # set limits for axes
    axes.plot(x,y,'ro-',linewidth=1)
    axes.set_xlabel("x"); axes.set_ylabel("y(x)")
    plt.grid()

"""Plotting of two input arrays (y1(x),y2(x)) sharing same borders for X axis but with different x values"""
def plotTwoArrDiffX(x1,x2,y1,y2,title:str="Samples",label1:str="y1",label2:str="y2"):
    plt.rc('font',family='serif') # Trying to use open-source font
    plt.figure(); axes = plt.axes(); plt.title(title)
    if (min(y1) > min(y2)):
        minY = min(y2)
    else:
        minY = min(y1)
    if (max(y1) < max(y2)):
        maxY = max(y2)
    else:
        maxY = max(y1)
    axes.set(xlim=(min(x1),max(x1)),ylim=(minY,maxY*1.05))  # set limits for axes
    axes.plot(x1,y1,'ro',linewidth = 2, label=label1)
    axes.plot(x2,y2,'b-',linewidth = 2,label=label2)
    axes.legend() # Necessary for displaying the legend
    plt.grid()