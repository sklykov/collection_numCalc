# -*- coding: utf-8 -*-
"""
Testing algoritms for lists.

@author: sklykov

"""
# Testing excluding the meshgrid coordinates from the overall meshgrid
a = [(i, j) for i in range(5) for j in range(5)]
i_pl = 1; j_pl = 3; h_o = 2; w_o = 3
b = [(i, j) for i in range(i_pl, i_pl+h_o) for j in range(j_pl, j_pl+w_o)]
c = a[:]
for exc_el in b:
    try:
        c.remove(exc_el)
    except ValueError:
        pass
