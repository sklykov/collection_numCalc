# coding=utf-8
"""
Verbose / detailed implementation of Sliding Mean Value (Smoothing) for 1D arrays.

@author: sklykov

@license: The Unlicense

"""
import numpy as np


def sliding_mean(x: np.ndarray, n_values: int = 3) -> np.ndarray:
    """
    Get array with averaged by sliding values.

    Parameters
    ----------
    x : np.ndarray
        1D array.
    n_values : int, optional
        Number of values for averaging. The default is 3.

    Returns
    -------
    np.ndarray
       1D array with legth = len(x) - n_values + 1 containing sliding average values.
    """
    # Edge cases checking - user input substition for meaningful value
    if n_values <= 1:
        n_values = 2
    if len(x) < n_values:
        return np.asarray(np.sum(x)/len(x))
    else:
        averaged = []; av = 0.0
        for i in range(len(x) - n_values + 1):
            if i == 0:
                av = np.sum(x[i:i+n_values])
            else:
                av += -x[i-1] + x[i+n_values-1]
            averaged.append(av)
        return np.asarray(averaged)/n_values


# %% Tests
x_vals2 = [1, 2, 3, 4]
x_smooth2 = np.round(sliding_mean(x_vals2))

x_vals = [0.02, 0.1, 0.19, 0.33, 0.41, 0.5, 0.62, 0.73, 0.79, 0.82, 0.91, 1.0]
x_smooth = np.round(sliding_mean(x_vals), 2)
