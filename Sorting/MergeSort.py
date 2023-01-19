# -*- coding: utf-8 -*-
"""
Merge Sort Demo (the top-bottom approach as far as I understand).

@author: sklykov
@license: The Unlicense
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
    ar1[i] = np.random.randint(-20, 21)
for i in range(len(ar2)):
    ar2[i] = np.random.randint(-1000, 2001)
# Generation of a presorted array
for i in range(len(ar3)):
    ar3[i] = i
# Generation of a presorted array with a reverse order - in theory, the worst case for sorting by Quick Sort
ar4 = np.array([len(ar4)-i for i in range(len(ar4))])


# %% Main function for performing Merge sorting
# As a possible TODO: re-implementation using np.array instead of lists for temporary values
def MergeSort(xIn, t0):
    if (isinstance(xIn, np.ndarray)) and (len(xIn) > 1):
        x = xIn.copy()  # Ok, again copy of the input array
        nSteps = len(xIn)//2  # Integer division for calculating integer number of steps, 5//2 = 2 (!)
        reminder = len(xIn) % 2  # Always 0 or 1 (an even or odd array size)
        i = 1  # For steps for output array construction
        lTemp = [[0, 0] for i in range(nSteps)]  # Formation of a temporary list with subarray pairs
        # Appending a final value to an array as single list, if array size is odd
        if reminder > 0:
            length = [x[len(x)-1]]
            lTemp.append(length)

        # Forming the sorted pairs
        for i in range(0, nSteps):  # nSteps - excluding
            if x[2*i] < x[2*i+1]:
                (lTemp[i])[0] = x[2*i]
                (lTemp[i])[1] = x[2*i+1]
            else:
                (lTemp[i])[1] = x[2*i]
                (lTemp[i])[0] = x[2*i+1]
#        print(lComposing,"pairs of values") # debugging of a programm

        x = np.array(PairsMerging(lTemp))  # Calling function for sorting pairs

        # Checking correctness of sorting + benchmarking
        isSortCorrect = CheckSortArray(x)
        t1 = time.process_time()  # Tick time
        t = round((t1-t0), 3)  # rounding to seconds for completion of the sorting operation
        return (x, t, isSortCorrect)

    else:
        print("Input parameter isn't the np.array or it's too small")
        return (0, 0, False)


# %% Recursive function for merging pairs formed at the first step
def PairsMerging(xIn: list):
    """
    Recursive function for merging pairs formed at the first step.

    Parameters
    ----------
    xIn : list
        xIn - input list containing pairs of subarrays for sorting.

    Returns
    -------
    Merged pair.

    """
#    print(xIn,"input array for Pairs Merging")  # Debugging

    # while loop below - instead of recursive call of this function -
    while (len(xIn) > 1):
        xOut = [None]*((len(xIn)//2) + (len(xIn) % 2))  # a temporary array for forming output
        sortingStep = 1  # Stepping and while() loop below going on pairs of subarrays for composing them in a sorted subarr

        # while loop below going on the pairs of subarrays for making composed, sorted subarray for an output
        while (sortingStep <= (len(xIn)//2)):
            xTemp = [i*0 for i in range(len(xIn[2*(sortingStep-1)])+len(xIn[2*(sortingStep-1)+1]))]  # For saving values
            # from pairs of lists to sort
            l1 = 0; l2 = 0  # Indexes for comparing values from both subarrays - donors for composing result sorted array
            iFill = 0  # for filling resulting sorted subarray - result of this recursive function

            # Picking values from two subarrays for making composed subarray as a result
            while (l1 < len(xIn[2*(sortingStep-1)])) and (l2 < len(xIn[2*(sortingStep-1)+1])):
                if (xIn[2*(sortingStep-1)])[l1] < (xIn[2*(sortingStep-1)+1])[l2]:
                    xTemp[iFill] = (xIn[2*(sortingStep-1)])[l1]; l1 += 1
                else:
                    xTemp[iFill] = (xIn[2*(sortingStep-1)+1])[l2]; l2 += 1
                iFill += 1

            # Adding below the remaining, last biggest value from two subarrays to a composed subarray (output of recursion)
            if (l1 < len(xIn[2*(sortingStep-1)])):
                while ((l1 < len(xIn[2*(sortingStep-1)]))):  # Adding remaining values from subarrays to a composed one
                    xTemp[iFill] = (xIn[2*(sortingStep-1)])[l1]; l1 += 1; iFill += 1
            elif (l2 < len(xIn[2*(sortingStep-1)+1])):
                while ((l2 < len(xIn[2*(sortingStep-1)+1]))):  # Adding remaining values from subarrays to a composed one
                    xTemp[iFill] = (xIn[2*(sortingStep-1)+1])[l2]; l2 += 1; iFill += 1

#            print(xTemp,"resulting of subarray")
            xOut[sortingStep-1] = xTemp
            sortingStep += 1

        # Adding odd value (a single value subarray) to a resulting subarray - an output one
        if (len(xIn) % 2) > 0:
            xOut[sortingStep-1] = xIn[len(xIn)-1]
        xIn = xOut.copy()

    # Final function result
    return xIn[0]
#        PairsMerging(xOut) # Recursive step for sorting - can't find the reason why in the end returning None in if clause


# %% Merge sorting testing and benchmarking
t0 = time.process_time()  # get the starting point from the CPU time [s]
(arSorted1, t, res) = MergeSort(ar1, t0)
print('Initial array for demo', ar1); print("Sorted array for demo", arSorted1)
t0 = time.process_time()  # get the starting point from the CPU time [s]
(arSorted2, t, res) = MergeSort(ar2, t0)
print("Sorting of an unsorted array takes s = ", t); print('Is sorting correct?', res)
t0 = time.process_time()  # get the starting point from the CPU time [s]
(arSorted3, t, res) = MergeSort(ar3, t0)
print("Sorting of a presorted array takes s = ", t)
t0 = time.process_time()  # get the starting point from the CPU time [s]
(arSorted4, t, res) = MergeSort(ar4, t0)
print("Sorting of a presorted in descending order array (the worst case) takes s = ", t)
