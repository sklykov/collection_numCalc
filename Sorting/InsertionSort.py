# -*- coding: utf-8 -*-
"""
Insertion Sort Demo
Developed in the Spyder IDE
@author: ssklykov
"""
# %% Import section
import numpy as np
import time
from IsSortingCorrect import CheckSortArray

# %% Generation of a random sequence with integer numbers
ar1 = np.zeros(8,dtype=int)  # Initialize empty (filled with zeros) numpy array for demo
ar2 = np.zeros(2000,dtype=int) # Initialize empty (filled with zeros) numpy array for benchmark
ar3 = np.zeros(1000,dtype=int) # Initialize empty (filled with zeros) PRESORTED numpy array for becnhmark

# Fill initialized nparray with random integers from the specified ranges
for i in range(len(ar1)):
    ar1[i] = np.random.randint(-10,21)
for i in range(len(ar2)):
    ar2[i] = np.random.randint(-1000,2001)
# Generation of presorted array
for i in range(len(ar3)):
    ar3[i] = i

# %% Insertion sort implementation in the following function
# Again, np.array - mutable, operations of sorting / insertion could be performed on it
def InsertSort(xIn,to):
    if (isinstance(xIn,np.ndarray)) and (len(xIn)>1):
        x = xIn.copy()  # Create copy of an input array
        xj = 0  # Interim values
        # Actual insertion sort
        for j in range(1,len(x)):
            xj = x[j] # holding "pivot"
            while((j-1) >= 0):
                if (x[j-1] > x[j]):
                    x[j] = x[j-1] # shift the bigger value on one step forward
                    x[j-1] = xj  # assign the pivot on the place of bigger value
                    j-=1 # decreasing the index for comparing pivot with previous value
                else: break  # break shifting, because the value before pivot already on the right place

        isSortCorrect = CheckSortArray(x)
        t1 = time.process_time() # Tick time
        t = round((t1-t0),3)  # rounding to seconds for completion of the sorting operation
        return (x,t,isSortCorrect)

    else:
        print("Input parameter isn't the np.array or it's too small")
        return (0,0,False)

# %% Insertion sorting and benchmarking of this operation
t0 = time.process_time()  # get the starting point from the CPU time [s]
(sortedAr1,t,res) = InsertSort(ar1,t0)
print('Initial array for demo',ar1); print("Sorted array for demo",sortedAr1)
t0 = time.process_time()  # get the starting point from the CPU time [s]
(sortedAr2,t,res) = InsertSort(ar2,t0)
print("Sorting of a unsorted array takes s = ",t); print('Is sorting correct?',res)
t0 = time.process_time()  # get the starting point from the CPU time [s]
(sortedAr3,t,res) = InsertSort(ar3,t0)
print("Sorting of a unsorted array takes s = ",t)