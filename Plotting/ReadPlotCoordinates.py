# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 2019
This script is saved for only keeping the gathered info and possible reusing it in future.
This script is created for performing simple reading and plotting using the matplotlib library. 
The main emphasis is on the formatting method of graphs 
The txt file for reading contains two columns of coordinates (int numbers)
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
        
        # Demo of using plot
        # Below many commands have beem written in "Matlab style". They have "OOP" alternative style...
        plt.figure(1) 
        plt.plot(x,y,'ro-',linewidth=2,markersize=11) # formatting line itself
        xMin = min(x); xMax = max(x); yMin = min(y); yMax = max(y)
        plt.axis([xMin-1,xMax+1,yMin-1,yMax+1]) # specifiying axis ranges in form xmin xmax
        plt.xlabel('x coordinates')
        plt.ylabel('y coordinates')
        plt.xticks(x)
        plt.yticks(np.arange(yMin,yMax+1,step=2))
        plt.grid() # the grid ON
        
          
    finally:
        # Guarantee of file closing 
        file.close()
        
else:
    print("Specify path to a file!")
