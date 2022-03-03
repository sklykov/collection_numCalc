# -*- coding: utf-8 -*-
"""
Simple GUI for representing of generated noisy image using PyQT.

@author: ssklykov
"""
# %% Imports
# import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout
import numpy as np
from generate_noise_pic import generate_noise_picture
from threading import Thread
# import matplotlib.pyplot as plt
import time
import pyqtgraph


# %% Class wrapper for threaded noisy picture generation
class ImageGenerator(Thread):
    height = 100; width = 100; image = np.zeros((height, width), dtype='uint8')

    def __init__(self, height: int = 100, width: int = 100):
        Thread.__init__(self); self.height = height; self.width = width

    def run(self):
        self.image = generate_noise_picture(self.height, self.width)




# %% Implementation of all windows inside the child class
class SimUscope(QMainWindow):
    # Class variables
    __flagGeneration = False

    def __init__(self, img_height, img_width):
        super().__init__()
        self.imageGenerator = ImageGenerator(img_height, img_width); self.img_height = img_height; self.img_width = img_width
        self.img = np.zeros((self.img_height, self.img_width), dtype='uint8')
        self.setWindowTitle("Simulation of uscope camera")
        self.setGeometry(200, 200, 600, 500)
        self.imageWidget = pyqtgraph.ImageView(parent=self); self.imageWidget.ui.roiBtn.hide(); self.imageWidget.ui.menuBtn.hide()
        self.imageWidget.setImage(self.img)
        self.button = QPushButton("Generate Single Pic", self); self.button.clicked.connect(self.generate_single_pic)
        self.vbox = QVBoxLayout(self.imageWidget); self.vbox.addWidget(self.button)
        self.setLayout(self.vbox)
        self.setCentralWidget(self.imageWidget)


    def generate_single_pic(self):
        self.imageGenerator.start(); self.imageGenerator.join(); self.imageWidget.setImage(self.imageGenerator.image)
        self.imageGenerator = ImageGenerator(self.img_height, self.img_width)


# %% Tests
if __name__ == "__main__":
    my_app = QApplication([])  # application without any command-line arguments
    my_app.setQuitOnLastWindowClosed(True)  # workaround for forcing the quit of the application window for returning to the kernel
    main_window = SimUscope(200, 200); main_window.show()
    # !!!: Don't forget that waiting that thread complete its task guarantees all proper assignments (below)
    imageGenerator = ImageGenerator(height=200, width=200); imageGenerator.start(); imageGenerator.join(); img = imageGenerator.image

    my_app.exec()
