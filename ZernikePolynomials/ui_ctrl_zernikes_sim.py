# -*- coding: utf-8 -*-
"""
GUI for representing and controlling Zernike polynomials.

@author: ssklykov
"""
# %% Imports
import tkinter as tk
from tkinter.ttk import Button, Frame, Label  # better looking buttons for tkinter, rewriting standard one from tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # import canvas container from matplotlib for tkinter
from zernike_pol_calc import get_plot_zps_polar

# %% Some constants
orders = [(-1, 1)]  # Y tilt

# %% GUI class
class ZernikeCtrlUI(Frame):  # all widgets master class - top level window

    master: tk.Tk
    figure = plt.figure(num=1, figsize=(4, 4), layout='tight'); plt.close(1)

    def __init__(self, master):
        global step_r, step_theta, orders
        super().__init__(master, width=800, height=500)  # some default initial values for width / height of the main window
        self.master.title("Zernike's polynomials controls and respresentation")
        # Widgets creation and specification
        self.testButton = Button(self, text="Test", command=self.plot_zernikes)
        self.zernikesLabel = Label(self, text="Zernike polynomials specification")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self); self.plotWidget = self.canvas.get_tk_widget()
        # Placing all created widgets in the grid layout
        self.testButton.grid(row=1, rowspan=1, column=7, columnspan=1, sticky='new')
        self.zernikesLabel.grid(row=0, rowspan=1, column=3, columnspan=2)
        self.plotWidget.grid(row=1, rowspan=4, column=1, columnspan=4)

        self.grid()
        self.grid_propagate(False)  # Preventing of shrinking of windows to conform with all placed widgets (for pack and grid)

    def plot_zernikes(self):
        self.figure = get_plot_zps_polar(self.figure, orders)
        print(self.figure)
        self.canvas.draw(); self.canvas.draw_idle()


# %% Launch section
if __name__ == "__main__":
    root = tk.Tk()  # running instance of Tk()
    ui_ctrls = ZernikeCtrlUI(root)  # construction of the main frame
    ui_ctrls.mainloop()
