# coding=utf-8
"""
Verbose / detailed implementation of Sliding Window concept and one of its application - max sum of subarray.

@author: sklykov

@license: The Unlicense

"""
# %% Imports
from numbers import Real


# %% Func.
def max_sum_subarray(array: list, subarray_len: int = 3) -> Real:
    """
    Find max sum between subarray with specified length in provided array (list).

    Parameters
    ----------
    array : list
        Input array with real numbers.
    subarray_len : int, optional
        Length of subarray for finding the max sum. The default is 3.

    Returns
    -------
    Real
        Max sum of all checked subarrays.

    """
    # Edge cases - subarray length is more than array length, array should be non-empty
    if len(array) == 0:
        return 0
    else:
        if len(array) <= subarray_len:
            return sum(array)
        else:
            # Normal case, length of a subarray smaller than an array's length
            sum_subarray = sum(array[0:subarray_len])  # initial sum of a subarray
            max_sum = sum_subarray  # initial value for the max sum
            for i in range(subarray_len, len(array)):
                sum_subarray += array[i] - array[i-subarray_len]
                max_sum = max(max_sum, sum_subarray)
            return max_sum



# %% Tests
if __name__ == "__main__":
    # Edge case 1
    array1 = []
    print("Max subarray with length 3 sum:", max_sum_subarray(array1), "| array:", array1)
    array2 = [1, 0.5]
    print("Max subarray with length 3 sum:", max_sum_subarray(array2), "| array:", array2)
    array3 = [1, 0.5, -0.5, 2, -1.5, 2.4, 1, 0.1, -1, -0.2, 2]
    print("Max subarray with length 3 sum:", max_sum_subarray(array3), "| array:", array3)
    array4 = [1, 0.5, -0.5, 2, -1.5, 2.4, 1, 0.1, -1, -0.2, 2]
    print("Max subarray with length 2 sum:", max_sum_subarray(array4, 2), "| array:", array4)
