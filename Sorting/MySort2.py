# -*- coding: utf-8 -*-
"""
Sorting "swap it and merge!"
Main steps:
    (1) swap values in founded pairs (e.g. [54]) or subarrays of unsorted values (e.g [864])
    (2) merge two presorted subarrays in sorted ones with bigger size - growing them until sorting done
(!): Still bad performance for randomly distributed values
Developed in the Spyder IDE using Kite
@author: ssklykov
"""
# %% Import section
import numpy as np
import time
from IsSortingCorrect import CheckSortArray

# %% Generation of a random sequence with integer numbers
ar1 = np.zeros(10, dtype=int)  # Initialize empty (filled with zeros) numpy array for demo
ar2 = np.zeros(2000, dtype=int)  # Initialize empty (filled with zeros) numpy array for benchmark
ar3 = np.zeros(1000, dtype=int)  # Initialize empty (filled with zeros) PRESORTED numpy array for becnhmark
ar4 = np.zeros(1000, dtype=int)  # Initialize empty (filled with zeros) PRESORTED (reverse order) numpy array for becnhmark

# Fill initialized nparray with random integers from the specified ranges
for i in range(len(ar1)):
    ar1[i] = np.random.randint(0, 11)
for i in range(len(ar2)):
    ar2[i] = np.random.randint(-1000, 2001)
# Generation of a presorted array
for i in range(len(ar3)):
    ar3[i] = i
# Generation of a presorted array with a reverse order - in theory, the worst case for sorting by Quick Sort
ar4 = np.array([len(ar4)-i for i in range(len(ar4))])


# %% Function for perform the sorting with type input checking
# NOTICE: np.array is mutable, thus the function creates the entire copy of an input array
def MySort(xIn, t0):
    if (isinstance(xIn, np.ndarray)) and (len(xIn) > 1):
        x = xIn.copy()  # Create copy of an input array
        isSortingPerformedCorrect = False  # initialization the value for returning

        # First step - presort found unsorted chunks into sorted ones
        # The algorithm looks for unsorted chuncks of original array
        k = 0  # indexing of elements in an array
        swaps = 0  # number of performed swaps during each sorting iteration on overall input array
        while (k < (len(x)-1)):
            l = 0  # counting pairs of unsorted values

            # Checking if there is unsorted subarrays presented
            if ((x[k]-x[k+1]) > 0):
                l += 1  # for counting how many further elements in unsorted sequence
                # Calculation of length of unsorted subsequence (like 8765..)
                while ((k+l+1) < len(x)) and ((x[k+l]-x[k+l+1]) > 0):  # (!): first condition should be tested firstly!
                    l += 1

            # Swapping value in unsorted parts of input array
            if (l > 0):
                xk = 0  # for swapping - interim value for exchanging two swapped values
                # Swapping unsorted pairs
                if (l == 1):
                    # print(x,"before pair swapping") # Debugging purposes
                    xk = x[k]; x[k] = x[k+1]; x[k+1] = xk; k += l
                    # print(x,"after pair swapping") # Debugging purposes
                # Swapping even numbers of unsorted pairs from a whole unsorted subarray (like [876])
                elif (l % 2 == 0):
                    # print(l % 2,"- l % 2 |",l, "l")
                    # print(x,"before even swapping") # Debugging purposes
                    n = 0  # for making right swapping
                    while (n < (l / 2)):
                        xk = x[k+n]; x[k+n] = x[k+l-n]; x[k+l-n] = xk; n += 1
                    # print(x,"after even swapping") # Debugging purposes
                    k += l
                # Swapping odd numbers of unsorted pairs from a whole unsorted subarray (like [8765])
                else:
                    # Here is found number of pairs in unsorted sequence is odd
                    # print(x,"before odd swapping") # Debugging purposes
                    n = 0  # for making right swapping
                    while (n < ((l // 2) + 1)): # l//2 - integer division, 5//2 = 2
                        xk = x[k+n]; x[k+n] = x[k+l-n]; x[k+l-n] = xk; n += 1
                    # print(x,"after odd swapping") # Debugging purposes
                    k += l
                swaps += 1
            k += 1  # Iteration step through all elements in an array

        # Second step - compose subarrays into sorted array
        # print(x,"presorted on 1st step array") # For debugging
        i = 0  # iteration through values in an array
        while (i < len(x)-1):
            # if unsorted part found - starting merging of presorted before subarrays in one overall array
            if (x[i] > x[i+1]):  # Condition of encountering border of two sorted chunks (1st step)
                j = i+1  # Starting accounting next border between next unsorted chunks
                while (j < len(x)-1) and (x[j] < x[j+1]):  # Defying such border
                    j += 1
                # Sorting two founded unsorted subarrays in one sorted by merging them into it
                l = 0; l1 = 0; l2 = i+1 # for stepping through two subarrays
                compArr = np.zeros(j+1, dtype=int)  # Weak programming practice but fast to implement - to use subarray here
                # Composing sorted subarray while one of two chunks runs out of elements for composing
                while ((l2 < (j+1)) and (l1 < (i+1))):
                    if ((x[l1] > x[l2]) and (l2 < j+1)):
                        compArr[l] = x[l2]; l += 1; l2 += 1  # step on the second subarray
                    elif ((x[l1] <= x[l2]) and (l1 < (i+1))):
                        compArr[l] = x[l1]; l += 1; l1 += 1  # step on the first subarray

                # if there are still elements in a first chunk, use them for composing sorted output subarray
                if (l1 < (i+1)):
                    while (l1 < (i+1)):
                        compArr[l] = x[l1]; l += 1; l1 += 1
                # the same but for the second chunk - use it for composition
                else:
                    while (l2 < (j+1)):
                        compArr[l] = x[l2]; l += 1; l2 += 1

                # print(j,"j"); print(compArr,"composed array") # For debugging

                # Transferring composed sorted subarray to the overall array
                if len(compArr) < len(x):
                    for k in range(0, j+1):
                        x[k] = compArr[k]
                else:
                    x = compArr
                # print(x,"after composing") # For debugging
                i = -1  # Start again the searching of unsorted chunks
            i += 1  # step over the entire array

        # Compose returning values
        isSortingPerformedCorrect = CheckSortArray(x)
        t1 = time.process_time()  # Tick time
        t = round((t1-t0), 3)  # rounding to seconds for completion of the sorting operation
        return (x, t, isSortingPerformedCorrect)

    else:
        print("Input parameter isn't the np.array or it's too small")
        return (0, 0, False)


# %% Implementation of the swapping sorting algorithm and benchmarking of this operation
t0 = time.process_time()  # get the starting point from the CPU time [s]
(sortedAr1, t, res) = MySort(ar1, t0)
print('Initial array for demo', ar1); print("Sorted array for demo ", sortedAr1)
t0 = time.process_time()  # get the starting point from the CPU time [s]
(arSorted2, t, res) = MySort(ar2, t0)
print("Sorting of an unsorted array takes s = ", t); print('Is sorting correct?', res)
t0 = time.process_time()  # get the starting point from the CPU time [s]
(arSorted3, t, res) = MySort(ar3, t0)
print("Sorting of a presorted array takes s = ", t)
t0 = time.process_time()  # get the starting point from the CPU time [s]
(arSorted4, t, res) = MySort(ar4, t0)
print("Sorting of a presorted in descending order array takes s = ", t)
