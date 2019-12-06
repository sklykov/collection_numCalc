# -*- coding: utf-8 -*-
"""
Attempt to make definition of sample algebraic functions more OOP
Using static methods for returning values without class instance creation and using any class attributes
Developed in the Spyder IDE
@author: ssklykov
"""
# Default constructor used...
class ExampleAlgF():
    @staticmethod
    def example1(x:float):
        return pow(x,3) - 2*pow(x,2) + x - 2
    @staticmethod
    def example2(x:float):
        return x**4 - 3*(x**3) - 5*(x**2) + 7*x + 5