# -*- coding: utf-8 -*-
"""
Verbose implementation of classical counting problem - get all char frequencies in a string.

@author: sklykov

@license: The Unlicense

"""
from collections import Counter


# Manual implementation of simple counter
def get_char_freqs(target: str) -> dict:
    """
    Return the dictionary containing frequencies of all characters found in a target string.

    Parameters
    ----------
    target : str
        Target string.

    Returns
    -------
    dict
        Chars as keys, frequencies as values.
    """
    freqs = {}
    if len(target) > 0:  # check for the simplest edge case
        for char in target:
            freqs[char] = freqs.get(char, 0) + 1  # if key isn't found - return 0 by default
    return freqs


# %% Test
if __name__ == "__main__":
    sample_text = ("Lorem ipsum dolor sit amet consectetur adipisicing elit. Deserunt explicabo quo consectetur voluptas laboriosam nam, "
                   + "eius saepe quidem sapiente eos debitis asperiores minima magni esse neque eaque quae fuga. Ipsum?")
    char_frequencies = get_char_freqs(sample_text)
    char_frequencies_counter = Counter(sample_text)  # convert to dict for observing identity in a variable viewer

    # Get min / max frequent chars from a manually calculated dict
    min_count = min(char_frequencies.values()); max_count = max(char_frequencies.values())
    chars_min_count = [k for k, v in char_frequencies.items() if v == min_count]
    chars_max_count = [k for k, v in char_frequencies.items() if v == max_count]

    # Single min / max frequent chars
    char_min_count = min(char_frequencies, key=char_frequencies.get)   # works because iterating over dict == iteration over keys
    char_min_count_counter = min(char_frequencies_counter, key=char_frequencies_counter.get)
