# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 2019
This script is saved for only keeping the gathered info and possible reusing it in future.
This script is created for performing simple reading and plotting using the matplotlib library.
The main emphasis is on the formatting method of graphs
The txt file for reading contains two columns of coordinates (int numbers)
The IDE for development - Spyder
Generally, much information has been collected on the StackOverflow website
@author: ssklykov
"""
import matplotlib.pyplot as plt
import numpy as np

path = "XY.txt" # Don't forget to specify actual path to file
if (path!= ""):
    try:
        # Reading data
        x = []; y = []  # maybe redundant initializers
        file = open(path, 'r')
        lines = file.readlines()
        for eachLine in lines:
            strBuff = eachLine.split(" ") # splitting two columns of numbers
            xBuff = strBuff[0]; yBuff = strBuff[1]
            x.append(int(xBuff))  # put variable in a list with X coordinates
            y.append(int(yBuff))

        # Below many commands have beem written in "Matlab style"
        # They have "OOP" alternative style but for many classes somehow autocompletion doesn't work!

        # Figure preformatting
        wSize = 1.2*(7.0/2.54); hSize = (7.0/2.54) # converting santimeters to inches
        plt.figure(figsize=(wSize,hSize))  # empty container with specified sizes
        axes = plt.axes() # get the instance of axes for making minor plots and suppress the poped warning
        plt.rc('font',family='serif') # setting font type
        plt.plot(x,y,'ro',markersize=9) # formatting line itself / auto adding axes
        xMin = min(x); xMax = max(x); yMin = min(y); yMax = max(y)
        plt.axis([xMin-1,xMax+1,yMin-2,yMax+2]) # specifiying axis ranges in form xmin xmax
        plt.xlabel('timepoints',fontsize=14,fontfamily='Liberation Serif')
        plt.ylabel('measurments',fontsize=14,fontfamily='Liberation Serif')
        # separation 'timepoints' to even and odd numbers
        x1=[]; x2=[]
        for i in range(len(x)):
            if (i % 2 == 0):
                x1.append(x[i])
            else:
                x2.append(x[i])
        # ticks handling
        plt.xticks(x1,fontsize=10,fontfamily='Liberation Serif') # main xticks
        axes.set_xticks(x2,minor=True)  # minor ticks
        plt.yticks(np.arange(yMin,yMax+1,step=4),fontsize=10,fontfamily='Liberation Serif')
        axes.set_yticks(np.arange(yMin+2,yMax-1,step=2),minor=True)

        plt.grid(which='both') # the grid ON
        # preparing for figure saving
        plt.tight_layout()  # fill the picture better
        # plt.savefig("SimpleLine.png",dpi=300)
        # plt.savefig("SimpleLine.jpg",dpi=300)

    finally:
        # Guarantee of file closing
        file.close()

else:
    print("Specify path to a file!")
