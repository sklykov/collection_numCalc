# -*- coding: utf-8 -*-
"""
Attempt to make definition of sample algebraic functions more OOP.

Using static methods for returning values without class instance creation and using any class attributes.

@author: sklykov
@license: The Unlicense
"""
import math


class ExampleAlgF():

    @staticmethod
    def example1(x: float):
        return pow(x, 3) - 2*pow(x, 2) + x - 2  # 1 real root: x = 2

    @staticmethod
    def example2(x: float):
        return pow(x, 4) - 3*pow(x, 3) - 5*pow(x, 2) + 7*x + 5  # 3 real roots: x ~= 1.45, x ~= -0.58, x ~= 3.74

    @staticmethod
    def example3(x: float):
        return math.exp(x) - 0.5*pow(x,2) - 25  # Just function that intersects with x=0 axes with high slope, x ~= 3.43

    @staticmethod
    def example3Derivative(x: float, epsilon: float):  # Specification of an approximation to derivative to 3rd example function
        return (ExampleAlgF.example3(x+epsilon)-ExampleAlgF.example3(x))/epsilon  # f'(x) ~= (f(x+dx)-f(x))/dx

    @staticmethod
    def approxDerivative(func, x2:float, x1:float):
        try:
            x = (func(x2)-func(x1))/(x2-x1)
        except ZeroDivisionError:
            x = 0
        return x
