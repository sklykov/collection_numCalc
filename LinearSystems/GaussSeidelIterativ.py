# -*- coding: utf-8 -*-
"""
Implementation of Gauss-Seidel iterative method of solving linear system like Ax = b
Developed in the Spyder IDE using Kite
@author: ssklykov
"""

#%% Import section
import numpy as np
import sys

#%% Parameters for modelling
epsilon = 0.01  # For defying precision of calculations and the condition for stopping iterations
nSize = 3  # matrix sizes
nDigits = 3 # precision of truncations

# %% Using class for avoiding type checking of input parameters for Gauss-Seidel method - generate all matricies
class TestingMatricies():
    nSize = 2
    def __init__(self,nSize:int=2):
        self.A = np.zeros((nSize,nSize),dtype=float)
        self.b = np.zeros((nSize,1),dtype=float)
        self.x = np.zeros((nSize,1),dtype=float)
        self.nSize = nSize
    def setA(self,A:list):
        self.A = np.array(A)
    def setB(self,b:list):
        for i in range(self.nSize):
            self.b[i] = b[i]

#%% Implementation of Gauss-Seidel algorithm
def GaussSeidel(testM:TestingMatricies,epsilon:float=0.01):
    flag = True # for returning from this method after checking convergence
    n = testM.nSize; A = testM.A; b = testM.b # Get the values
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
            print("Warning: diagonal elements aren't dominant"); break
        i += 1

    # Gauss-Seidel iterations
    # Calculation of a modified matricies S and t for making iterations x(k) = S*x(k-1) + t
    S = A.copy(); t = b.copy()
    for i in range(n):
        t[i] = b[i]/A[i][i]
        for j in range(n):
            S[i][j] = A[i][j]/A[i][i]


#%% Testing the implemented algorithms
testM = TestingMatricies(nSize) # generate initial matricies
testM.setA([[3,2,-1],[2,-2,4],[-1,0.5,-1]]) # setting a matrix A
testM.setB([1,-2,0]) # setting a matrix b
GaussSeidel(testM,epsilon)