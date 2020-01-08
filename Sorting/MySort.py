# -*- coding: utf-8 -*-
"""
Another modified sort (not sure that actual "bubble" but looks similiar)
Sorting "swap it!"
Main idea - to swap values in founded pairs (e.g. [54]) or subarrays of unsorted values (e.g [864])
This is slow algorithm right now because the ineffective pairs swapping step comes after first effective step of sorting all
unsorted subarrays. Now, the sorting of unsorted array takes longer time than even the bubble sort algorithm
TODO: increase speed of sorting after first effective and fast step! Avoid pairs swapping!
Developed in the Spyder IDE using Kite
@author: ssklykov
"""
# %% Import section
import numpy as np
import time
from IsSortingCorrect import CheckSortArray

# %% Generation of a random sequence with integer numbers
ar1 = np.zeros(10,dtype=int)  # Initialize empty (filled with zeros) numpy array for demo
ar2 = np.zeros(2000,dtype=int) # Initialize empty (filled with zeros) numpy array for benchmark
ar3 = np.zeros(1000,dtype=int) # Initialize empty (filled with zeros) PRESORTED numpy array for becnhmark
ar4 = np.zeros(1000,dtype=int) # Initialize empty (filled with zeros) PRESORTED (reverse order) numpy array for becnhmark

# Fill initialized nparray with random integers from the specified ranges
for i in range(len(ar1)):
    ar1[i] = np.random.randint(0,11)
for i in range(len(ar2)):
    ar2[i] = np.random.randint(-1000,2001)
# Generation of a presorted array
for i in range(len(ar3)):
    ar3[i] = i
# Generation of a presorted array with a reverse order - in theory, the worst case for sorting by Quick Sort
ar4 = np.array([len(ar4)-i for i in range(len(ar4))])

# %% Function for perform the sorting with type input checking
# NOTICE: np.array is mutable, thus the function creates the entire copy of an input array
def MySort(xIn,t0):
    if (isinstance(xIn,np.ndarray)) and (len(xIn)>1):
        x = xIn.copy()  # Create copy of an input array
        flag = True; isSortingPerformedCorrect = False  # Interim values

        # while below on each iteration trying to find unsorted chunks and if found performing swapping
        while(flag):
            # The algorithm looks for unsorted chuncks of original array
            k = 0 # indexing of elements in an array
            swaps = 0  # number of performed swaps during each sorting iteration on overall input array
            while (k < (len(x)-1)):
                l = 0 # counting pairs of unsorted values

                # Checking if there is unsorted subarrays presented
                if ((x[k]-x[k+1]) > 0):
                    l += 1 # for counting how many further elements in unsorted sequence
                    # Calculation of length of unsorted subsequence (like 8765..)
                    while ((k+l+1) < len(x)) and ((x[k+l]-x[k+l+1]) > 0): # (!): first condition should be tested firstly!
                        l += 1

                # Swapping value in unsorted parts of input array
                if (l > 0):
                    xk = 0 # for swapping - interim value for exchanging two swapped values
                    # Swapping unsorted pairs
                    if (l == 1):
                        # print(x,"before pair swapping") # Debugging purposes
                        xk = x[k]; x[k] = x[k+1]; x[k+1] = xk; k += l-1
                        # print(x,"after pair swapping") # Debugging purposes
                    # Swapping even numbers of unsorted pairs from a whole unsorted subarray (like [876])
                    elif (l % 2 == 0):
                        # print(l % 2,"- l % 2 |",l, "l")
                        # print(x,"before even swapping") # Debugging purposes
                        n = 0 # for making right swapping
                        while (n < (l / 2)):
                            xk = x[k+n]; x[k+n] = x[k+l-n]; x[k+l-n] = xk; n += 1
                        # print(x,"after even swapping") # Debugging purposes
                        k += l
                    # Swapping odd numbers of unsorted pairs from a whole unsorted subarray (like [8765])
                    else:
                        # Here is found number of pairs in unsorted sequence is odd
                        # print(x,"before odd swapping") # Debugging purposes
                        n = 0 # for making right swapping
                        while (n < ((l // 2) + 1)): # l//2 - integer division, 5//2 = 2
                            xk = x[k+n]; x[k+n] = x[k+l-n]; x[k+l-n] = xk; n += 1
                        # print(x,"after odd swapping") # Debugging purposes
                        k += l
                    swaps += 1

                # print(k,"current index")
                k += 1 # Iteration step through all elements in an array

            # Checking if there is still presented any unsorted chunk
            if (swaps > 0):
                # print("number of performed swaps:",swaps) # Debugging
                flag = True
            else:
                flag = False
            # flag = False # For making only single step

        isSortingPerformedCorrect = CheckSortArray(x)
        t1 = time.process_time() # Tick time
        t = round((t1-t0),3)  # rounding to seconds for completion of the sorting operation
        return (x,t,isSortingPerformedCorrect)

    else:
        print("Input parameter isn't the np.array or it's too small")
        return (0,0,False)

# %% Implementation of the swapping sorting algorithm and benchmarking of this operation
t0 = time.process_time()  # get the starting point from the CPU time [s]
(sortedAr1,t,res) = MySort(ar1,t0)
print('Initial array for demo',ar1); print("Sorted array for demo ",sortedAr1)
t0 = time.process_time()  # get the starting point from the CPU time [s]
(arSorted2,t,res) = MySort(ar2,t0)
print("Sorting of an unsorted array takes s = ",t); print('Is sorting correct?',res)
t0 = time.process_time()  # get the starting point from the CPU time [s]
(arSorted3,t,res) = MySort(ar3,t0)
print("Sorting of a presorted array takes s = ",t)
t0 = time.process_time()  # get the starting point from the CPU time [s]
(arSorted4,t,res) = MySort(ar4,t0)
print("Sorting of a presorted in descending order array takes s = ",t)