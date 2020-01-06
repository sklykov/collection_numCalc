# -*- coding: utf-8 -*-
"""
Another modified bubble sort - with first step of presorting large unsorted sequences
Developed in the Spyder IDE using Kite
@author: ssklykov
"""
# %% Import section
import numpy as np
import time
from IsSortingCorrect import CheckSortArray
from matplotlib import pyplot as plt

# %% Generation of a random sequence with integer numbers
ar1 = np.zeros(9,dtype=int)  # Initialize empty (filled with zeros) numpy array for demo

# Fill initialized nparray with random integers from the specified ranges
for i in range(len(ar1)):
    ar1[i] = np.random.randint(0,9)

# %% Function for perform the sorting with type input checking
# NOTICE: np.array is mutable, thus the function creates the entire copy of an input array
def ModBubbleSort(xIn,t0):
    if (isinstance(xIn,np.ndarray)) and (len(xIn)>1):
        x = xIn.copy()  # Create copy of an input array
        xj = 0; i = 0; flag = True; isSortingPerformedCorrect = False  # Interim values
        # First step - finding large sequence of unsorted values
        k = 0 # indexing of elements in an array
        while (k < (len(x)-1)):
            l = 0 # counting pairs of unsorted values
            if ((x[k]-x[k+1]) > 0):
                l += 1 # for counting how many further elements in unsorted sequence
                # while ((x[k+l]-x[k+l+1]) < 0) and (k+l+1 < len(x)):
                #     l += 1
            if (l > 0):
                xk = 0 # for swapping
                if (l == 1):
                    xk = x[k]; x[k] = x[k+1]; x[k+1] = xk; k += l
            k += 1
        print(x)

        # Actual bubble sort
        isSortingPerformedCorrect = CheckSortArray(x)
        t1 = time.process_time() # Tick time
        t = round((t1-t0),3)  # rounding to seconds for completion of the sorting operation
        return (x,t,isSortingPerformedCorrect)

    else:
        print("Input parameter isn't the np.array or it's too small")
        return (0,0,False)

# %% Modified Bubble sorting and benchmarking of this operation
t0 = time.process_time()  # get the starting point from the CPU time [s]
(sortedAr1,t,res) = ModBubbleSort(ar1,t0)
print('Initial array for demo',ar1); print("Sorted array for demo ",sortedAr1)