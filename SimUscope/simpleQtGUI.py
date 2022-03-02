# -*- coding: utf-8 -*-
"""
Simple GUI for representing of generated noisy image using PyQT.

@author: ssklykov
"""
# %% Imports
# import sys
from PyQt5.QtWidgets import QApplication, QWidget

# %% Tests
my_app = QApplication([])  # application without any command-line arguments
my_app.setQuitOnLastWindowClosed(True)  # workaround for forcing the quit of the application window for returning to the kernel
root = QWidget(); root.show()
my_app.exec()




