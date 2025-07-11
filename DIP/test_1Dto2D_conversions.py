# -*- coding: utf-8 -*-
"""
Simple comparison of 1D to 2D array transform (used for conversion raw camera pixels data to an image).

@author: sklykov

"""
import numpy as np
import time
import random

# Generation of random pixel values in 1D list
w = 4112; h = 3008  # specific to high resolution camera
a = [0]*w*h
for i in range(len(a)):
    a[i] = random.randint(0, 254)
img = np.zeros((w, h))

# Manual conversion into 2D image
t1 = time.perf_counter()
i = 0
for j in range(h):
    img[:, j] = a[i*w:(i+1)*w]
    i += 1
print(f"\nManual conversion from 1D to 2D {w}x{h} array takes {int(round(1E3*(time.perf_counter() - t1), 0))} ms")

# Conversion using numpy function
t1 = time.perf_counter()
img2 = np.reshape(np.asarray(a), (w, h))
print(f"\nNumpy reshaping from 1D to 2D {w}x{h} array takes {int(round(1E3*(time.perf_counter() - t1), 0))} ms")
