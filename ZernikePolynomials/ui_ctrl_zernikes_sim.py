# -*- coding: utf-8 -*-
"""
GUI for representing and controlling Zernike polynomials.

@author: ssklykov
"""
# %% Imports
import tkinter as tk
# Below: themed buttons for tkinter, rewriting standard one from tk
from tkinter.ttk import Button, Frame, Label, OptionMenu, Checkbutton
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # import canvas container from matplotlib for tkinter
from zernike_pol_calc import get_plot_zps_polar, get_classical_polynomial_name
import matplotlib.figure as plot_figure
import time
import numpy as np

# %% Some constants


# %% GUI class
class ZernikeCtrlUI(Frame):  # all widgets master class - top level window
    """Class specified the GUI controlling instance."""

    master: tk.Tk
    figure: plt.figure
    plotColorbar: bool; flagFlattened: bool
    amplitudes: list; orders: list; changedSliders: int

    def __init__(self, master):
        # Values initialization
        super().__init__(master)  # initialize the main window (frame) for all widgets
        self.plotColorbar = False
        self.master.title("Zernike's polynomials controls and respresentation")
        master.geometry("+15+50")  # put the main window on the (+x, +y) coordinate away from the top left monitor coordinate
        self.amplitudes = [0.0, 0.0]  # default amplitudes for the 1st order
        self.orders = [(-1, 1), (1, 1)]; self.flagFlattened = False; self.changedSliders = 1
        self.amplitudes_sliders_dict = {}
        # Widgets creation and specification (almost all - buttons)
        self.refreshPlotButton = Button(self, text="Refresh Plot", command=self.plot_zernikes)
        self.zernikesLabel = Label(self, text="Zernike's polynomials controls up to: ")
        self.figure = plot_figure.Figure(figsize=(5, 5))  # Default empty figure for
        self.canvas = FigureCanvasTkAgg(self.figure, master=self); self.plotWidget = self.canvas.get_tk_widget()
        # !!! Below - the way of how associate tkinter buttons with the variables and their states! THEY ARE DECOUPLED!
        self.varPlotColorbarButton = tk.BooleanVar(); self.varPlotColorbarButton.set(False)
        self.plotColorbarButton = Checkbutton(self, text="Colorbar", command=self.colorBarPlotting, onvalue=True, offvalue=False,
                                              variable=self.varPlotColorbarButton)
        self.flattenButton = Button(self, text="Flatten all", command=self.flattenAll)
        # Below - specification of OptionMenu from ttk for polynomials order selection, fixed 1st value not shown => StackOverflow
        self.order_n = ["1st ", "2nd ", "3rd ", "4th ", "5th ", "6th ", "7th "]
        self.order_list = [item + "order" for item in self.order_n]
        self.clickable_list = tk.StringVar(); self.clickable_list.set(self.order_list[0])
        self.max_order_selector = OptionMenu(self, self.clickable_list, self.order_list[0], *self.order_list,
                                             command=self.numberOrdersChanged)
        # Specificiation of two case selectors: Simulation / Controlling DPP
        self.listDevices = ["Pure Simulator", "DPP + simulator"]; self.device_selector = tk.StringVar()
        self.device_selector.set(self.listDevices[0]); self.deviceSelectorButton = OptionMenu(self, self.device_selector, *self.listDevices)
        # Below - additional window for holding the sliders with the amplitudes
        self.ampl_ctrls = tk.Toplevel(master=self)  # additional window, master - the main window
        self.ampl_ctrls.geometry("+680+50")   # put this additional window with some offset for the representing it next to the main one
        self.ampl_ctrls.wm_transient(self)  # Seems that it makes accessible the buttons values from the this window to the main
        self.ampl_ctrls.title("Amplitude controls"); self.ampl_ctrls.protocol("WM_DELETE_WINDOW", self.no_exit)
        # Placing all created widgets in the grid layout
        self.zernikesLabel.grid(row=0, rowspan=1, column=0, columnspan=1)
        self.max_order_selector.grid(row=0, rowspan=1, column=1, columnspan=1)
        self.refreshPlotButton.grid(row=0, rowspan=1, column=2, columnspan=1)
        self.plotColorbarButton.grid(row=0, rowspan=1, column=3, columnspan=1)
        self.flattenButton.grid(row=0, rowspan=1, column=4, columnspan=1)
        self.deviceSelectorButton.grid(row=7, rowspan=1, column=0, columnspan=1)
        self.plotWidget.grid(row=1, rowspan=6, column=0, columnspan=6)

        self.grid()
        # self.grid_propagate(False)  # Preventing of shrinking of windows to conform with all placed widgets (for pack and grid)
        # Another approach - to pack all the buttons
        # self.plotWidget.pack(side=tk.BOTTOM); self.z_11.pack(side=tk.RIGHT); self.z11.pack(side=tk.RIGHT)
        # self.zernikesLabel.pack(side=tk.LEFT); self.max_order_selector.pack(side=tk.LEFT); self.refreshPlotButton.pack(side=tk.LEFT)
        # self.pack()
        self.plot_zernikes()  # initial flat image
        # set the value for the OptionMenu and call function for construction of ctrls
        self.clickable_list.set(self.order_list[3]); self.after(0, self.numberOrdersChanged(self.order_list[3]))

    def plot_zernikes(self):
        """
        Plot the sum of specified Zernike's polynomials amplitudes.

        Returns
        -------
        None.

        """
        t1 = time.time()
        # below: update the plot
        self.figure = get_plot_zps_polar(self.figure, orders=self.orders, step_r=0.005, step_theta=0.8,
                                         alpha_coefficients=self.amplitudes, show_amplitudes=self.plotColorbar)
        # t3 = time.time(); print("plot time(ms):", int(np.round((t3-t1)*1000, 0)))
        self.canvas.draw()  # redraw the figure
        t2 = time.time(); print("redraw time(ms):", int(np.round((t2-t1)*1000, 0)))

    def colorBarPlotting(self):
        """
        Redraw of colormap with the Zernike's polynomials sum on the unit radius aperture.

        Returns
        -------
        None.

        """
        self.plotColorbar = self.varPlotColorbarButton.get(); self.plot_zernikes()

    def sliderValueChanged(self, new_pos):
        """
        Any slider value has been changed and this function handles it.

        Parameters
        ----------
        new_pos : double
            It is sent by the evoking of this event button.

        Returns
        -------
        None.

        """
        # new_pos sent by the associated button
        i = 0
        for key in self.amplitudes_sliders_dict.keys():
            # print(self.amplitudes_sliders_dict[key].get(), end=" ")  # FOR DEBUG
            self.amplitudes[i] = self.amplitudes_sliders_dict[key].get(); i += 1
        # print("Recorded amplitudes:", self.amplitudes)  # FOR
        if self.changedSliders > 1:  # more than one slider changed => flatten operation
            self.changedSliders -= 1
        if not self.flagFlattened:  # if no flatten flag, redraw the plot
            self.plot_zernikes()
        else:
            if self.changedSliders == 1:  # if all sliders finally zeroed, redraw the plot
                self.flagFlattened = False; self.plot_zernikes()

    def flattenAll(self):
        """
        Make all amplitude sliders controls equal to 0.0 value.

        Returns
        -------
        None.

        """
        self.flagFlattened = True  # flag for preventing the redrawing
        for key in self.amplitudes_sliders_dict.keys():
            if abs(self.amplitudes_sliders_dict[key].get()) > 0.0001:  # if not actually equal to zero
                self.amplitudes_sliders_dict[key].set(0.0)
                self.changedSliders += 1  # counting number of zeroed sliders

    def numberOrdersChanged(self, selected_order: str):
        """
        Handle the event of order specification.

        Parameters
        ----------
        selected_order : str
            Reported string with the selected order.

        Returns
        -------
        None.

        """
        # print(selected_order)  # FOR DEBUG
        n_orders = int(selected_order[0])
        # Refresh the TopLevel window and the associated dictionary with buttons
        self.ampl_ctrls.destroy(); self.ampl_ctrls = tk.Toplevel(master=self)
        self.ampl_ctrls.wm_transient(self); self.ampl_ctrls.protocol("WM_DELETE_WINDOW", self.no_exit)
        self.ampl_ctrls.title("Amplitude controls"); self.ampl_ctrls.geometry("+645+50")
        # Get the (m, n) values from the order specification
        self.orders = []; self.amplitudes_sliders_dict = {}; self.amplitudes = []  # refresh the associated controls and values
        for order in range(1, n_orders + 1):  # going through all specified orders
            m = -order  # azimutal order
            n = order  # radial order
            for polynomial in range(order + 1):  # number of polynomials = order + 1
                self.orders.append((m, n))  # store the values as tuples
                # Below - initialization of sliders for controlling amplitudes of Zernike's polynomials
                classical_name = get_classical_polynomial_name((m, n), short_names=True)
                self.amplitudes_sliders_dict[(m, n)] = tk.Scale(self.ampl_ctrls, from_=-1.0, to=1.0, orient='horizontal',
                                                                resolution=0.05, sliderlength=20, label=(f"Z{(m ,n)} " + classical_name),
                                                                tickinterval=0.5, length=152, command=self.sliderValueChanged,
                                                                repeatinterval=220)
                # self.amplitudes_sliders_dict[(m, n)].pack(side=tk.TOP)  # simplest way of packing - adding buttons on top of each other
                self.amplitudes.append(0.0)  # assign all zeros as the flat field
                m += 2  # according to the specification
        # Placing the sliders on the window
        for order in range(1, n_orders + 1):
            # rows_offset = n_orders - order
            rows_offset = 0
            m = -order  # azimutal order
            row_cursor = 0
            n = order  # radial order
            for polynomial in range(order + 1):  # number of polynomials = order + 1
                # self.amplitudes_sliders_dict[(m, n)].grid(row=(rows_offset + row_cursor), rowspan=1, column=(order-1), columnspan=1)
                self.amplitudes_sliders_dict[(m, n)].grid(row=(order-1), rowspan=1, column=(rows_offset + row_cursor), columnspan=1)
                m += 2; row_cursor += 1
        self.plot_zernikes()  # refresh the plot, not retain any values
        # print(self.orders)

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
