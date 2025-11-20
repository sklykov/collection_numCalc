# coding=utf-8
"""
Verbose / detailed implementation of Recursion and basic Dynamic Programming concepts.

In simple words: Dynamic programming is a caching previous results for preventing their recomputations.

@author: sklykov

@license: The Unlicense

"""
# %% Imports
import math


# %% Parameters

# %% Func-s
def classic_factorial(n: int) -> int:
    """
    Classic factorial example using recursion.

    Parameters
    ----------
    n : int
        Factorial n!.

    Returns
    -------
    int
        n!

    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return n*classic_factorial(n-1)


def fibonacci_down_top(n: int) -> int:
    """
    Use memoization of previous results for each step as dynamic programming approach.

    Parameters
    ----------
    n : int
        Index of Fibonacci number.

    Returns
    -------
    int
        Fibonacci number.

    """
    if n < 0:
        return 0
    if n <= 1:  # works for both 0 and 1
        return n
    # Code below is reachable only if conditions above not resolved
    memoized_results = [0]*(n+1)  # storing previously calculated values for each step
    memoized_results[1] = 1
    for i in range(2, n+1):  # sequentially calculate next value using previously calculated from #2
        memoized_results[i] = memoized_results[i-2] + memoized_results[i-1]
    return memoized_results[i]


def fibonacci_top_down(n: int, results_dict: dict = {}) -> int:
    """
    Use top -> bottom memoization and recursion for calculation of Fibonacci numbers.

    Parameters
    ----------
    n : int
        index of a number.
    results_dict : dict, optional
        For memoization of results. The default is {}.

    Returns
    -------
    int
        Fibonacci number.

    """
    if n in results_dict:  # automatically checks n in dict keys
        return results_dict[n]
    if n < 0:
        return 0
    if n <= 1:
        return n
    results_dict[n] = fibonacci_top_down(n-2, results_dict) + fibonacci_top_down(n-1, results_dict)
    return results_dict[n]


# %% Tests
if __name__ == "__main__":
    k = 5
    print("Recursion factorial:", classic_factorial(k), "| From Math library:", math.factorial(k))
    k = 10
    print("Recursion factorial:", classic_factorial(k), "| From Math library:", math.factorial(k))
    k = 15
    print("Recursion factorial:", classic_factorial(k), "| From Math library:", math.factorial(k))

    print("*****************************************************")

    m = 5
    print(f"Fibonacci bottom-up #{m}:", fibonacci_down_top(m), "| top-bottom:", fibonacci_top_down(m), "| Tabular:", 5)
    m = 12
    print(f"Fibonacci bottom-up #{m}:", fibonacci_down_top(m), "| top-bottom:", fibonacci_top_down(m), "| Tabular:", 144)
    m = 19
    print(f"Fibonacci bottom-up #{m}:", fibonacci_down_top(m), "| top-bottom:", fibonacci_top_down(m), "| Tabular:", 4181)
