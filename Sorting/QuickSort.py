# -*- coding: utf-8 -*-
"""
Quick Sort Demo
Developed in the Spyder IDE
@author: ssklykov
"""
# %% Import section
import numpy as np
import time
from IsSortingCorrect import CheckSortArray

# %% Generation of a random sequence with integer numbers
ar1 = np.zeros(8,dtype=int)  # Initialize empty (filled with zeros) numpy array for demo
#ar2 = np.zeros(2000,dtype=int) # Initialize empty (filled with zeros) numpy array for benchmark
#ar3 = np.zeros(1000,dtype=int) # Initialize empty (filled with zeros) PRESORTED numpy array for becnhmark

# Fill initialized nparray with random integers from the specified ranges
for i in range(len(ar1)):
    ar1[i] = np.random.randint(-10,21)
#for i in range(len(ar2)):
#    ar2[i] = np.random.randint(-1000,2001)
## Generation of presorted array
#for i in range(len(ar3)):
#    ar3[i] = i

# %% Quick sort implementation
def QuickSort(xIn,t0):
    if (isinstance(xIn,np.ndarray)) and (len(xIn)>1):
        x = xIn.copy()  # Create copy of an input array

        # TODO: actual implementation

        isSortCorrect = CheckSortArray(x)
        t1 = time.process_time() # Tick time
        t = round((t1-t0),3)  # rounding to seconds for completion of the sorting operation
        return (x,t,isSortCorrect)

    else:
        print("Input parameter isn't the np.array or it's too small")
        return (0,0,False)

# %% Quick sorting and benchmarking of this operation
t0 = time.process_time()  # get the starting point from the CPU time [s]

