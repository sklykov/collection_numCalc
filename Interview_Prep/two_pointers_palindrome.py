# coding=utf-8
"""
Verbose / detailed implementation of Two Pointers concept and one of its application - checking if a word is a palindrome.

@author: sklykov

@license: The Unlicense

"""
# %% Imports


# %% Func.
def is_palindrome(word: str) -> bool:
    """
    Check that input word (not sentence!) is palindrome.

    Parameters
    ----------
    word : str
        Single word containing characters.

    Returns
    -------
    bool
        If word is palindrome (same writing from start to end and from end to start).

    """
    # Edge cases (assuming that word contains only characters, not entire sentence)
    if len(word) == 0:
        return False
    elif len(word) == 1:
        return True
    # Normal case, word has >= 2 chars
    else:
        word = word.lower()  # prevent any edge case with first capital letter
        i, j = 0, len(word) - 1  # two pointers
        while i < j:  # compare in the loop first character and last one going to the center
            if word[i] != word[j]:
                return False
            i += 1; j -= 1
        return True


# %% Tests
if __name__ == "__main__":
    word0 = ''
    print(f"Word '{word0}' is palindrome:", is_palindrome(word0))
    word1 = 'b'
    print(f"Word '{word1}' is palindrome:", is_palindrome(word1))
    word2 = 'aa'
    print(f"Word '{word2}' is palindrome:", is_palindrome(word2))
    word3 = 'abba'
    print(f"Word '{word3}' is palindrome:", is_palindrome(word3))
    word4 = 'abgracadabra'
    print(f"Word '{word4}' is palindrome:", is_palindrome(word4))
    word5 = 'cababac'
    print(f"Word '{word5}' is palindrome:", is_palindrome(word5))
