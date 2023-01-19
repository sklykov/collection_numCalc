# -*- coding: utf-8 -*-
"""
Checking the correctness of sorting.

Return True if input array is sorted correctly (in ascending order).
"""
# %% Import section
import numpy as np


# %% Checking results of sorting (by comparing all values in a sorted array)
def CheckSortArray(xSortedIn):
    if (isinstance(xSortedIn, np.ndarray)) and (len(xSortedIn) > 1):
        for i in range(0, len(xSortedIn)-1):
            if (xSortedIn[i+1] < xSortedIn[i]):
                return False  # Checking the order in pairs
        return True
    else:
        return False
