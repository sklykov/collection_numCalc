# -*- coding: utf-8 -*-
"""
Examples of data manipulation.

@author: sklykov

@license: The Unlicense

"""
from collections import deque

lst = [x*x for x in range(1, 201)]
dq = deque(lst)

# Right end append / remove element: both O(1) complexity for list and deque
el = (len(lst)+1)*(len(lst)+1)
lst.append(el);  dq.append(el)   # O(1)
lst.pop();  dq.pop()  # 0(1)

# Left end (start) append / remove element: O(1) - deque, O(n) - list
lst.insert(0, 0); lst.pop(0)  # both - O(n)
dq.appendleft(0); dq.popleft()  # both - O(1)
