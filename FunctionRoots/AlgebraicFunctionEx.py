# -*- coding: utf-8 -*-
"""
A few examples of algebraic function specification.

It's a function returning value of function (f(x)), where x - an input value.
"""
import math


def fVal(x: float, example: int = 1):  # 1st example - default
    if (example == 1):
        return x**3 - 2*(x**2) + x - 2  # 1 real root: x = 2
    elif (example == 2):
        return x**4 - 3*(x**3) - 5*(x**2) + 7*x + 5  # 3 real roots: x ~= 1.45, x ~= -0.58, x ~= 3.74
    elif (example == 3):
        return math.exp(x) - 0.5*pow(x, 2) - 25
    else:
        return 0
