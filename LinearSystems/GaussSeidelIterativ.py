# -*- coding: utf-8 -*-
"""
Implementation of Gauss-Seidel iterative method of solving linear system like Ax = b
Convergence (stability) of this method - ?
Developed in the Spyder IDE using Kite
@author: ssklykov
"""

#%% Import section
import numpy as np
import sys

#%% Parameters for modelling
epsilon = 0.01  # For defying precision of calculations and the condition for stopping iterations
nSize = 4  # matrix sizes
nDigits = 3 # precision of truncations
nMaxIterations = 4 # number of maximum iterations (preventing infinite loops )

# %% Using class for avoiding type checking of input parameters for Gauss-Seidel method - generate all matricies
class TestingMatricies():
    nSize = 2
    def __init__(self,nSize:int=2):
        self.A = np.zeros((nSize,nSize),dtype=float)
        self.b = np.zeros(nSize,dtype=float)
        self.x = np.zeros(nSize,dtype=float)
        self.nSize = nSize
    def setA(self,A:list):
        nRows = len(A) # number of rows
        nCols = len(A[0]) # number of columns
        if (nRows == nCols) and (nRows == self.nSize):
            self.A = (np.array(A)).astype(dtype=float) # Important: carefully use conversions (type of an input array - int)
    def setB(self,b:list):
        nRows = len(b)
        if (nRows == self.nSize): # Checking input array
            self.b = (np.array(b)).astype(dtype=float)


#%% Implementation of Gauss-Seidel algorithm
"""Iterative for linear system like Ax = b, there A and b - matricies which should be assigned before call this method"""
def GaussSeidel(testM:TestingMatricies,epsilon:float=0.01,nDigits:int=3,nMaxIterations:int=10):
    flag = True # for returning from this method after checking convergence
    n = testM.nSize; A = testM.A; b = testM.b; x = testM.x # Get the values
    # simple check if input matricies allow the Gauss-Seidelalgorithm to converge
    # 1st check - all diagonal elements are non-zero (it's possible to modify this algorithm to handle this case)
    i = 0
    while i < n:
        if (A[i][i] == 0):
            flag = False; break # Finishing checking because already some element == 0
        i += 1
    if (not flag):
        print("One of diagonal element a[i][i] == 0, please reorder equations in the input system")
        return None
    # 2nd check - diagonal dominance of elements in a matrix A (not absolutely needed but preferable)
    maxNonDiag = sys.float_info.min
    i = 0; j = 0
    while i < n:
        while j < n:
            if (i != j) and (A[i][j] > maxNonDiag):
                maxNonDiag = A[i][j]
            j += 1
        i += 1
    # print(maxNonDiag) # Debugging
    i = 0
    while i < n:
        if (abs(A[i][i]) < maxNonDiag):
            print("Warning: diagonal elements aren't dominant, this method may diverge"); break
        i += 1

    # Gauss-Seidel iterations
    # Calculation of a modified matricies S and t for making iterations x(k) = S*x(k-1) + t
    S = A.copy(); t = b.copy()
    for i in range(n):
        t[i] = b[i]/A[i][i]
        for j in range(n):
            S[i][j] = -A[i][j]/A[i][i]
    # print(A)
    # print(t)
    # print(S)

    # Performing the iterations
    for i in range(n): # starting value for performing iterative process (0th iteration)
        x[i] = t.item(i)
    # print(x) # Debugging
    delta = x.copy(); l = 1 # number of iterations + delta - corrections between two iterations
    while (l < nMaxIterations) and (StopIterations(delta,n,epsilon)):
        for i in range(0,n):
            sumCurrIter = 0.0 # sum using roots from the current iteration (k)
            if (i > 0):  # Condition for using already calculated iterations (k)
                for j in range(0,i):
                    sumCurrIter += S[i][j]*x[j]
            sumPrevIter = 0.0 # sum using roots from the previous iteration (k-1)
            for j in range(i,n):
                sumPrevIter += S[i][j]*x[j]
            # print(sumCurrIter,"current iteration contributions")
            # print(sumPrevIter,"previous iterations contributions")
            delta[i] =  sumCurrIter + sumPrevIter + t.item(i) # correction of current roots calculation
            x[i] += delta.item(i) # correction of roots (x1,x2...) - an actual iteration step
            # x[i] = round(x.item(i),nDigits) # introducing truncation error but allowing to control each iteration
        l += 1 # accounting a number of iterations for calculation of all roots (k)
        # print(delta)
        # print(x) # Debugging

    # Rounding final result - more robust than rounding each iteration
    for i in range(n):
        x[i] = round(x.item(i),nDigits)
    # Returning results of Gauss-Seidel method
    return x

# %% Checking the absolute differences of corrections and epsilon - returning True if
def StopIterations(delta,nSize:int,epsilon:float):
    flag = False # not continuing iterations in the method above
    for i in range(nSize):
        deltaS = round(abs(delta.item(i)),nDigits) # Introducing rounding for reducing complexety of comparison with epsilon
        if deltaS > epsilon:
            flag = True; break # the single coincidence is enough
    return flag

#%% Testing the implemented algorithms (good example)
testM = TestingMatricies(nSize) # generate initial matricies
testM.setA([[10,-1,2,0],[-1,11,-1,3],[2,-1,10,-1],[0,3,-1,8]]) # setting a matrix A
testM.setB([6,25,-11,15]) # setting a matrix b
xRoots = GaussSeidel(testM,epsilon,nDigits,nMaxIterations)
print("solution to the system:\n",xRoots)

# %% Testing the implemented algorithms (random example from the Wiki)
nSize = 3; nDigits = 3; nMaxIterations = 80
testM = TestingMatricies(nSize) # generate initial matricies
testM.setA([[3,2,-1],[2,-2,4],[-1,0.5,-1]]) # setting a matrix A
testM.setB([1,-2,0]) # setting a matrix b
xRoots = GaussSeidel(testM,epsilon,nDigits,nMaxIterations)
print("solution to the system:\n",xRoots)
print("The solution diverges")