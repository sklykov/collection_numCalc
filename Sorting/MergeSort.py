# -*- coding: utf-8 -*-
"""
Merge Sort Demo (the top-bottom approach as far as I understand)
Developed in the Spyder IDE
@author: ssklykov
"""
# %% Import section
import numpy as np
import time
from IsSortingCorrect import CheckSortArray

# %% Generation of a random sequence with integer numbers
ar1 = np.zeros(5,dtype=int)  # Initialize empty (filled with zeros) numpy array for demo

# Fill initialized nparray with random integers from the specified ranges
for i in range(len(ar1)):
    ar1[i] = np.random.randint(-20,21)

# %% Main function for performing Merge sorting
def MergeSort(xIn,t0):
    if (isinstance(xIn,np.ndarray)) and (len(xIn)>1):
        x = xIn.copy()  # Ok, again copy of the input array
        nSteps = len(xIn)//2 # Integer division for calculating integer number of steps, 5//2 = 2 (!)
        reminder = len(xIn)%2  # Always 0 or 1 (an even or odd array size)
        i = 1 # For steps for output array construction
        lComposing = [None]*(nSteps+reminder) # Initialization of a temporary list for composing results
        while (i <= nSteps):
            # Actually, below is a temporary hint - decribing explicitly the first step of a merge sorting
            if (i == 1):
                # Formation of a temporary list with subarray pairs
                lTemp = [[0,0] for i in range(nSteps)]  # Pairs of values for sorting
                # Appending a final value to an array as single list, if array size is odd
                if reminder > 0:
                    l = [x[len(x)-1]]
                    lTemp.append(l)
                # As a possible TODO: re-implementation using np.array instead of lists for temporary values
                # Forming the sorted pairs
                for i in range(0,nSteps):  # nSteps - excluding
                    if x[2*i] < x[2*i+1]:
                        (lTemp[i])[0] = x[2*i]; (lTemp[i])[1] = x[2*i+1]
                    else:
                        (lTemp[i])[1] = x[2*i]; (lTemp[i])[0] = x[2*i+1]
                lComposing = lTemp.copy() # Mimicking in and out operation
                print(lComposing)
#            else:
#                print(PairsMerging(lComposing),"result!")

            i += 1 # next sorting step

        # Checking correctness of sorting + benchmarking
        isSortCorrect = CheckSortArray(x)
        t1 = time.process_time() # Tick time
        t = round((t1-t0),3)  # rounding to seconds for completion of the sorting operation
        return (lComposing,t,isSortCorrect)

    else:
        print("Input parameter isn't the np.array or it's too small")
        return (0,0,False)

# %% Recursive function for merging pairs formed at the first step
def PairsMerging(xIn:list):
    print(xIn,"input array for Pairs Merging")
    if (len(xIn) <= 1):
        return xIn
    else:
        xOut = [[None]*((len(xIn)//2) + (len(xIn) % 2))] # a temporary array for forming output
#        print(xOut)
        sortingStep = 1
        while (sortingStep <= (len(xIn)//2)):
            xTemp = [i*0 for i in range(len(xIn[2*(sortingStep-1)])+len(xIn[2*(sortingStep-1)+1]))] # For saving values
            # from pairs of lists to sort
            print(xTemp)
            l1 = 0; l2 = 0 # Indexes for comparing values from both values
            while l1 < len:
                print((xIn[2*(sortingStep-1)])[l1],"input subarray")
                print((xIn[2*(sortingStep-1)+1])[l2],"input subarray")
                if (xIn[2*(sortingStep-1)])[l1] < (xIn[2*(sortingStep-1)+1])[l2]:
                    xTemp[i] = (xIn[2*(sortingStep-1)])[l1]; l1 += 1
                else:
                    xTemp[i] = (xIn[2*(sortingStep-1)+1])[l2]; l2 += 1

            sortingStep += 1

# %% Merge sorting testing and benchmarking
t0 = time.process_time()  # get the starting point from the CPU time [s]
(arSorted1,t,res) = MergeSort(ar1,t0)
#print('Initial array for demo',ar1); print("Sorted array for demo",arSorted1)