# -*- coding: utf-8 -*-
"""
Implementation of a simple linear congruental generator

@author: ssklykov
"""

# %% Import section
import numpy as np
import matplotlib.pyplot as plt

# %% The algorithm implementation
def simpleLCG(a:int=1,c:int=1,mod:int=2**31,n:int=10,seed:float=0):
    """Algorithm should return an array of float numbers potentially (depending on input parameters) distributing within [0,1)"""
    if ((mod <= 0) or (a <= 0) or (a > mod) or (c <= 0) or (c > mod) or (n > mod) or (seed < 0) or (seed > 1)):
        print("one or more input parameters is invalid")
        return None
    else:
        x = np.zeros(n,dtype=int); x[0] = seed
        xRand = np.zeros(n,dtype=float); xRand[0] = seed
        for i in range(0,n-1):
            x[i+1] = (a*x[i] + c) % mod
            xRand[i+1] = x[i+1] / mod
        return xRand


# Testing the implemented algorithm
a = 5; c = 1; mod = 10e6; n = 1000; seed = 0
xRand = simpleLCG(a,c,mod,n,seed)
(counts,bins) = np.histogram(xRand,10,[0,1])

# demonstrate historgram
plt.figure()
plt.hist(xRand)