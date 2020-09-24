# -*- coding: utf-8 -*-
"""
Solution of a system of linear equations with tridiagonal matrix
Developed in the Spyder IDE
@author: ssklykov
"""


def Solution(a, b, c, d):
    for i in range(1, len(d)):  # Range always not included the last value
        # Below the literal implementation from the book
        a[i] /= b[i-1]  # This is the substitution of a[i] values by alpha[i] = a[i]/b[i-1]
        b[i] -= a[i]*c[i-1]  # -//- by b[i] - alpha[i]*c[i-1]
        d[i] -= a[i]*d[i-1]  # this is "y" values - solution to the actual system Ly = d, there L composed from a,b,c
    d[len(d)-1] /= b[len(b)-1]  # backward substitution
    for i in range(len(d)-2, -1):
        d[i] = (d[i] - c[i]*d[i+1])/b[i]
    return d  # solution to the input system Gx = d
