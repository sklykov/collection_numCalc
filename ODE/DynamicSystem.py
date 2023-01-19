# -*- coding: utf-8 -*-
"""
Class for modelling dynamic systems - for further solving by the 2nd order solvers.

Dynamic system in such case - the 1D oscillator.

@author: sklykov
@license: The Unlicense
"""
import math


class dynamicSys():
    """The header class for composing all properties related to the dynamic system modelling."""

    def __init__(self, typeSys: str):
        self.typeSys = typeSys

    def externalForce(self, amplitude: float, x: float, radius: float = 1, time: float = 0, freq: float = 0):
        if (self.typeSys == "1st example"):
            if (x < -radius):
                return amplitude*math.exp(-abs(x+radius))
            elif ((x >= -radius) and (x <= 0)):
                return amplitude*math.pow((x/radius), 2)
            elif ((x > 0) and (x <= radius)):
                return -amplitude*math.pow((x/radius), 2)
            else:
                return -amplitude*math.exp(-abs(x-radius))
        else:
            return None

    def getForce(self, amplitudeExtForce: float, x: float, radius: float, mass: float,
                 gamma: float, speed: float, time: float = 1.0):
        if (self.typeSys == "1st example"):
            f = 0.0
            f = self.externalForce(amplitudeExtForce, x, radius) - gamma*speed
            try:
                f /= mass
            except ZeroDivisionError:
                print("Input mass is zero! Force divided by 1 as a default value")
                f /= 1
            return f
        else:
            return None
