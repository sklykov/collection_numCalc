# -*- coding: utf-8 -*-
"""
Bubble Simple Sort Demo
Developed in Spyder IDE
@author: ssklykov
"""
# %% Import section
import numpy as np
import time

# %% Generation of a random sequence with integer numbers
ar1 = np.zeros(5,dtype=int)  # Initialize empty (filled with zeros) numpy array for demo
ar2 = np.zeros(1000,dtype=int) # Initialize empty (filled with zeros) numpy array for benchmark
# Fill initialized nparray with random integers from the specified ranges
for i in range(len(ar1)):
    ar1[i] = np.random.randint(0,11)
for i in range(len(ar2)):
    ar2[i] = np.random.randint(0,101)

# %% Function for perform the sorting with type input checking
# NOTICE: np.array is mutable (!)
def BubbleSort(x):
    if (isinstance(x,np.ndarray)) and (len(x)>1):
        print('Initial array for demo',x)
        xj=0  # Interim value
        # Actual bubble sort
        for j in range(len(x)-1): # Pair sorting (1 attempt)
            if (x[j]>x[j+1]):
                xj = x[j]; x[j]=x[j+1]; x[j+1]=xj;
        print('1 step for Sorting',x)

    else:
        print("Input parameter isn't the np.array or it's too small")

# %% Bubble sorting and timing of operation
t0 = time.process_time()  # get the starting point from the CPU time [s]


# %%
BubbleSort(ar1)
