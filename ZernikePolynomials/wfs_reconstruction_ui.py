# -*- coding: utf-8 -*-
"""
GUI for perform wavefront reconstruction.

It's can be used for test of implemented algorithm (refernces in the corresponding scripts) of
many recorded images, stored locally.
@author: ssklykov
"""
# %% Imports
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # import canvas container from matplotlib for tkinter
import matplotlib.figure as plot_figure
# import time
import numpy as np


# %% Reconstructor GUI
class ReconstructionUI(tk.Frame):  # The way of making the ui as the child of Frame class - from official tkinter docs

    def __init__(self, master):
        super().__init__(master)  # initialize the toplevel widget
        self.master.title("UI for reconstruction of recorded wavefronts")


# %% Main launch
if __name__ == "__main__":
    rootTk = tk.Tk()  # toplevel widget of Tk there the class - child of tk.Frame is embedded
    reconstructor_gui = ReconstructionUI(rootTk)
    reconstructor_gui.mainloop()
