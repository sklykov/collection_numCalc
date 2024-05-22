# -*- coding: utf-8 -*-
"""
Useful notes and samples for the multiprocessing tasks met in other projects.

@author: @sklykov

"""

from math import cos, sin
from multiprocessing import Pool
from functools import partial
import time
import numpy as np


# %% Using Pool.map for processing values in a list with some fixed parameter:
def calc_func(x: float, y: float):
    """
    Calculate sin((x-y)*(x+y)) - cos((x+y)*(x+y)).

    Parameters
    ----------
    x : float
        x in radians.
    y : TYPE
        y in radians..

    Returns
    -------
    float
        Some complex function.

    """
    return round(sin((x-y)*(x+y)) - cos((x+y)*(x+y)), 3)


if __name__ == "__main__":
    # Calculation using list comprehension
    t1 = time.perf_counter()
    results_list_compr = [calc_func(round(0.15*i, 3), 0.5) for i in range(-5000, 5000, 1)]
    print("Calc. time for 'list comprehension' form ms:", int(round(1000*(time.perf_counter() - t1), 0)))

    # Calculation using Pool
    t1 = time.perf_counter()
    calc_f_fixed_arg = partial(calc_func, y=0.5)  # Note that first argument should be iterable, others - could be fixed with constants
    arg_x = [round(0.15*i, 3) for i in range(-5000, 5000, 1)]  # iterable on that can be performed for loop
    with Pool(4) as p:
        results_pool_map = p.map(calc_f_fixed_arg, arg_x)
    print("Calc. time for 'Pool().map(...)' form ms:", int(round(1000*(time.perf_counter() - t1), 0)))

    results_diff = np.asarray(results_pool_map) - np.asarray(results_list_compr)

    # !!! NOTE (1):
    # To share function definition across multiple python processes, it is necessary to rely on a serialization protocol.
    # The standard protocol in python is pickle but its default implementation in the standard library has several limitations.
    # For instance, it cannot serialize functions which are defined interactively or in the __main__ module.
    # Source: https://joblib.readthedocs.io/en/stable/parallel.html

    # !!! NOTE (2):
    # After running this script, it reveals that Pool is less performant than simple list comprehension

    # !!! NOTE (3):
    # Almost the same results for the joblib library. Seems only huge tasks can be really optimized
