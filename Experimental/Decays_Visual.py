# -*- coding: utf-8 -*-
"""
Visualization of decays laws.

@author: sklykov
@license: The Unlicense
"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt

# %% Demos
x = np.arange(1.1, 5, 0.01)
y1 = 1/np.sqrt(x-1)
yN = np.max(y1)
y2 = 1/(yN*(x-1))
y3 = 4*np.exp(-(x)/5)

plt.figure(figsize=(7, 7))
plt.plot(x, y1, label="y1")
plt.plot(x, y2, label="y2")
plt.plot(x, y3, label="y2")
plt.legend()
plt.tight_layout()
