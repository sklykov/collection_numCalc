# coding=utf-8
"""
Verbose / detailed implementation of fast retrieiving / searching of value using a hash map implementation - dict.

One of its application - counting frequency of characters in a string.

@author: sklykov

@license: The Unlicense

"""
# %% Imports


# %% Func.
def char_frequency(text: str, character: str) -> int:
    """
    Find frequency of a input character in a text.

    Note that if character not presented in a text, it will return 0.

    Parameters
    ----------
    text : str
        Input text as a string.
    character : str
        Single character (not checked as an edge case - that it is really single!)

    Returns
    -------
    int
        Frequency of a character in a text.

    """
    # Some edge cases first
    if len(text) == 0:
        return 0
    else:
        if len(character) == 0:
            return 0
        else:
            text = text.lower(); character = character.lower()
            # Edge case - character not in a text
            if character not in text:
                return 0
            # Normal case - character presented in a text, check the frequency
            else:
                freq_counts = {}  # could be cashed in a wrapper function for the same text and fast lookup for a new character
                for letter in text:
                    freq_counts[letter] = freq_counts.get(letter, 0) + 1  # get method for a dict with a default value
                return freq_counts[character]





# %% Tests
if __name__ == "__main__":
    sample_text = "abracadabra abba one two free"; char = 'a'
    print(f"Frequency of character '{char}' in text '{sample_text}':", char_frequency(sample_text, char))
    sample_text = ("Lorem ipsum dolor sit amet consectetur adipisicing elit. Deserunt explicabo quo consectetur voluptas laboriosam nam, "
                   + "eius saepe quidem sapiente eos debitis asperiores minima magni esse neque eaque quae fuga. Ipsum?")
    char = 'e'
    print(f"Frequency of character '{char}' in text '{sample_text}':", char_frequency(sample_text, char))
    char = 'D'; sample_text = 'aafagfgsapqpg,gs gageL::d;salkqwp12 21kjskfj210g 0ksa;lkfakl'  # only 1 appearance
    print(f"Frequency of character '{char}' in text '{sample_text}':", char_frequency(sample_text, char))
