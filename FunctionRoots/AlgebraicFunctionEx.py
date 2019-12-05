# -*- coding: utf-8 -*-
"""
A few examples of algebraic function specification
It's a function returning value of function (f(x)), where x - an input value
"""
def fVal(x:float,example:int=1):  # 1st example - default
    if (example==1):
        return x**3 - 2*(x**2) + x -2
    else:
        return 0

