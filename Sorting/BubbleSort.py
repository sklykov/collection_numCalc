# -*- coding: utf-8 -*-
"""
Bubble Simple Sort Demo
Developed in the Spyder IDE
@author: ssklykov
"""
# %% Import section
import numpy as np
import time
from IsSortingCorrect import CheckSortArray
# %% Generation of a random sequence with integer numbers
ar1 = np.zeros(9,dtype=int)  # Initialize empty (filled with zeros) numpy array for demo
ar2 = np.zeros(2000,dtype=int) # Initialize empty (filled with zeros) numpy array for benchmark
ar3 = np.zeros(1000,dtype=int) # Initialize empty (filled with zeros) PRESORTED numpy array for becnhmark

# Fill initialized nparray with random integers from the specified ranges
for i in range(len(ar1)):
    ar1[i] = np.random.randint(-50,51)
for i in range(len(ar2)):
    ar2[i] = np.random.randint(-1000,1001)
# Generation of presorted array
for i in range(len(ar3)):
    ar3[i] = i

# %% Function for perform the sorting with type input checking
# NOTICE: np.array is mutable, thus the function creates the entire copy of an input array
def BubbleSort(xIn,t0):
    if (isinstance(xIn,np.ndarray)) and (len(xIn)>1):
        x = xIn.copy()  # Create copy of an input array
        xj = 0; i = 0; flag = True; isSortingPerformedCorrect = False  # Interim values
        # Actual bubble sort
        while ((len(x)-1-i) >= 1) and flag:
            flag = False  # flag for checking if any swapping occured
            for j in range(0,len(x)-1-i): # Sorting in pairs, thus len(x)-1 - not included in a range
                if (x[j]>x[j+1]): # Swapping two values in a pair
                    xj = x[j]; x[j]=x[j+1]; x[j+1]=xj; flag = True;  # The flag = at least 1 swap occured, continue sorting
            i+=1; # Next step in sorting
        isSortingPerformedCorrect = CheckSortArray(x)
        t1 = time.process_time() # Tick time
        t = round((t1-t0),3)  # rounding to seconds for completion of the sorting operation
        return (x,t,isSortingPerformedCorrect)

    else:
        print("Input parameter isn't the np.array or it's too small")
        return (0,0,False)

# %% Bubble sorting and benchmarking of this operation
t0 = time.process_time()  # get the starting point from the CPU time [s]
(sortedAr1,t,res) = BubbleSort(ar1,t0)
print('Initial array for demo',ar1); print("Sorted array for demo",sortedAr1)
t0 = time.process_time()  # get the starting point from the CPU time [s]
(sortedAr2,t,res) = BubbleSort(ar2,t0)
print("Sorting of a unsorted array takes s = ",t); print("Is sorting correct?",res)
t0 = time.process_time()  # get the starting point from the CPU time [s]
(sortedAr3,t,res) = BubbleSort(ar3,t0)
print("Sorting of a presorted array takes s = ",t)