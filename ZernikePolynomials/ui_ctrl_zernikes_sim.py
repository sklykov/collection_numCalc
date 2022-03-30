# -*- coding: utf-8 -*-
"""
GUI for representing and controlling Zernike polynomials.

@author: ssklykov
"""
# %% Imports
import tkinter as tk
# Below: themed buttons for tkinter, rewriting standard one from tk
from tkinter.ttk import Button, Frame, Label, OptionMenu
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # import canvas container from matplotlib for tkinter
from zernike_pol_calc import get_plot_zps_polar
import matplotlib.figure as plot_figure
import time
import numpy as np

# %% Some constants
orders = [(0, 2)]  # Zernike orders


# %% GUI class
class ZernikeCtrlUI(Frame):  # all widgets master class - top level window
    """Class specified the GUI controlling instance."""

    master: tk.Tk
    figure = plt.figure(num=1, figsize=(4, 4), layout='tight'); plt.close(1)

    def __init__(self, master):
        global orders
        super().__init__(master)  # initialize the main window (frame) for all widgets
        self.plotColobar = False
        self.master.title("Zernike's polynomials controls and respresentation")
        master.geometry("+100+100")
        # Widgets creation and specification
        self.refreshPlotButton = Button(self, text="Refresh Plot", command=self.plot_zernikes)
        self.zernikesLabel = Label(self, text="Zernike's polynomials controls up to: ")
        self.figure = plot_figure.Figure(figsize=(5, 5))  # Default empty figure for
        self.canvas = FigureCanvasTkAgg(self.figure, master=self); self.plotWidget = self.canvas.get_tk_widget()
        self.order_n = ["1st ", "2nd ", "3rd ", "4th ", "5th ", "6th ", "7th "]
        self.order_list = [item + "order" for item in self.order_n]
        self.clickable_list = tk.StringVar(); self.clickable_list.set(self.order_list[0])
        self.plotColorbarButton = Button(self, text="Colorbar", command=self.colorBarPlotting)
        # Below - specification of OptionMenu from ttk, fixed 1st value not shown => StackOverflow
        self.max_order_selector = OptionMenu(self, self.clickable_list, self.order_list[0], *self.order_list)
        # Below - additional window for holding the sliders with the amplitudes
        self.ampl_ctrls = tk.Toplevel(master=self)  # additional window, master - the main window
        self.ampl_ctrls.geometry("+800+100")   # put this additional window with some offset for the representing it next to the main one
        self.ampl_ctrls.wm_transient(self)  # ???
        self.ampl_ctrls.title("Amplitude controls"); self.ampl_ctrls.protocol("WM_DELETE_WINDOW", self.no_exit)
        # Sliders for specification of Zernike's amplitudes controls
        self.z_11 = tk.Scale(self.ampl_ctrls, from_=-1.0, to=1.0, orient='horizontal', resolution=0.05, sliderlength=20, label="Z(-1,1)",
                             tickinterval=0.5, length=200)
        self.z11 = tk.Scale(self.ampl_ctrls, from_=-1.0, to=1.0, orient='horizontal', resolution=0.05, sliderlength=20, label="Z(1,1)",
                            tickinterval=0.5, length=200)
        # Placing all created widgets in the grid layout
        self.zernikesLabel.grid(row=0, rowspan=1, column=0, columnspan=1)
        self.max_order_selector.grid(row=0, rowspan=1, column=1, columnspan=1)
        self.refreshPlotButton.grid(row=0, rowspan=1, column=2, columnspan=1)
        self.plotColorbarButton.grid(row=0, rowspan=1, column=3, columnspan=1)
        self.plotWidget.grid(row=1, rowspan=5, column=0, columnspan=5)
        # self.z_11.grid(row=1, rowspan=1, column=10, columnspan=1)  # was to the main window
        # self.z11.grid(row=2, rowspan=1, column=10, columnspan=1)  # was to the main window
        self.z_11.pack(side=tk.TOP); self.z11.pack(side=tk.TOP)
        self.grid()
        # Another approach - to pack all the buttons
        # self.plotWidget.pack(side=tk.BOTTOM); self.z_11.pack(side=tk.RIGHT); self.z11.pack(side=tk.RIGHT)
        # self.zernikesLabel.pack(side=tk.LEFT); self.max_order_selector.pack(side=tk.LEFT); self.refreshPlotButton.pack(side=tk.LEFT)
        # self.pack()

        # Layout specification
        # self.grid_propagate(False)  # Preventing of shrinking of windows to conform with all placed widgets (for pack and grid)

    def plot_zernikes(self):
        """
        Plot the sum of specified Zernike's polynomials amplitudes.

        Returns
        -------
        None.

        """
        t1 = time.time()
        self.figure = get_plot_zps_polar(self.figure, orders, show_amplitudes=False)  # update the plot
        # t3 = time.time(); print("plot time(ms):", int(np.round((t3-t1)*1000, 0)))
        self.canvas.draw()  # redraw the figure
        t2 = time.time(); print("redraw time(ms):", int(np.round((t2-t1)*1000, 0)))

    def colorBarPlotting(self):
        self.plotColobar = not(self.plotColobar)
        print(self.plotColobar)
        if self.plotColobar:
            self.plotColorbarButton.style('pressed')
        else:
            self.plotColorbarButton.style()

    def no_exit(self):
        """
        Prevent to close the top level window with all the sliders for controlling amplitudes.

        Returns
        -------
        None.

        """
        pass


# %% Functions


# %% Launch section
if __name__ == "__main__":
    root = tk.Tk()  # running instance of Tk()
    ui_ctrls = ZernikeCtrlUI(root)  # construction of the main frame
    ui_ctrls.mainloop()
