# -*- coding: utf-8 -*-
"""
Quick Sort Demo (picking a final element in an array as a pivot, thus it's not optimal implementation!)
Developed in the Spyder IDE
@author: ssklykov
"""
# %% Import section
import numpy as np
import time
from IsSortingCorrect import CheckSortArray

# %% Generation of a random sequence with integer numbers
ar1 = np.zeros(8, dtype=int)  # Initialize empty (filled with zeros) numpy array for demo
ar2 = np.zeros(2000, dtype=int)  # Initialize empty (filled with zeros) numpy array for benchmark
ar3 = np.zeros(1000, dtype=int)  # Initialize empty (filled with zeros) PRESORTED numpy array for becnhmark
ar4 = np.zeros(1000, dtype=int)  # Initialize empty (filled with zeros) PRESORTED (reverse order) numpy array for becnhmark

# Fill initialized nparray with random integers from the specified ranges
for i in range(len(ar1)):
    ar1[i] = np.random.randint(-10, 21)
for i in range(len(ar2)):
    ar2[i] = np.random.randint(-1000, 2001)
# Generation of a presorted array
for i in range(len(ar3)):
    ar3[i] = i
# Generation of a presorted array with a reverse order - in theory, the worst case for sorting by Quick Sort
ar4 = np.array([len(ar4)-i for i in range(len(ar4))])


# %% Wrapping function for performing Quick Sort with checking of sorting results and timing (benchmarking)
def QuickSort(xIn, t0) -> tuple:
    if (isinstance(xIn, np.ndarray)) and (len(xIn) > 1):
        x = xIn.copy()  # Ok, again copy of the input array

        SortPivot(x, 0, len(x)-1) # Actual sorting step - see below

        isSortCorrect = CheckSortArray(x)
        t1 = time.process_time()  # Tick time
        t = round((t1-t0), 3)  # rounding to seconds for completion of the sorting operation
        return (x, t, isSortCorrect)

    else:
        print("Input parameter isn't the np.array or it's too small")
        return (0, 0, False)


# %% Function for recursive calling
def SortPivot(x, iStart, iFinal):
    """
    Implementation of Quick Sort algorithm that calls itself recursively for sorting at each step two subarrays:
    less than pivot and equal or more than pivot (pivot here is selected as a last value in subarray)
    input parameters: x - an array to sort, iStart and iFinal - indexes of subarray what actually sorted
    """
    iBorder = iStart  # Pivot - a base value for comparison, iBorder - a border between < and >= subarrays
    if (iStart >= iFinal):
        return
    else:
        # Actual dividing of an input array to "<" than pivot subarray and ">="
        for i in range(iStart, iFinal):
            if (x[i] < x[iFinal]) and (i != iBorder):
                temp = x[iBorder]; x[iBorder] = x[i]; x[i] = temp  # Swapping the found value with value that stayed before
                # and greater than compared element (i.e. element for swapping is greater than pivot (x[iBorder])
                iBorder += 1  # Move index of border on 1 step more further
            elif (x[i] < x[iFinal]) and (i == iBorder):  # additional condition for the case than elements in an array
                # are less than pivot and already "pre-sorted" for this position
                iBorder += 1
        # Swap element to place pivot in the right place (between "<" and ">=" subarrays)
        temp = x[iBorder]; x[iBorder] = x[iFinal]; x[iFinal] = temp
        # Actual recursive call
        SortPivot(x, iStart, iBorder-1)  # sorting of a "<" subarray
        SortPivot(x, iBorder+1, iFinal)  # sorting of a ">=" subarray


# %% Quick sorting and benchmarking of this operation
t0 = time.process_time()  # get the starting point from the CPU time [s]
(arSorted1, t, res) = QuickSort(ar1, t0)
print('Initial array for demo', ar1); print("Sorted array for demo", arSorted1)
t0 = time.process_time()  # get the starting point from the CPU time [s]
(arSorted2, t, res) = QuickSort(ar2, t0)
print("Sorting of an unsorted array takes s = ", t); print('Is sorting correct?', res)
t0 = time.process_time()  # get the starting point from the CPU time [s]
(arSorted3, t, res) = QuickSort(ar3, t0)
print("Sorting of a presorted array takes s = ", t)
t0 = time.process_time()  # get the starting point from the CPU time [s]
(arSorted4, t, res) = QuickSort(ar4, t0)
print("Sorting of a presorted array in descending order (the worst case) takes s = ", t)
