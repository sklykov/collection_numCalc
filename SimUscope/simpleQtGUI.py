# -*- coding: utf-8 -*-
"""
Simple GUI for representing of generated noisy image using PyQT.

@author: ssklykov
"""
# %% Imports
# import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QGridLayout
import numpy as np
from generate_noise_pic import generate_noise_picture
from threading import Thread
# import matplotlib.pyplot as plt
import time
import pyqtgraph

# %% Global variables as simple method for synchronization
flag_generation = False


# %% Class wrapper for threaded noisy single picture generation
class SingleImageGenerator(Thread):
    """Threaded class for generation of single noisy image."""

    height = 100; width = 100; image = np.zeros((height, width), dtype='uint8')

    def __init__(self, height: int = 100, width: int = 100):
        Thread.__init__(self); self.height = height; self.width = width

    def run(self):
        """
        Generate of single image and storing it inside the class variable.

        Returns
        -------
        None.

        """
        self.image = generate_noise_picture(self.height, self.width)


# %% Class wrapper for threaded noisy continuous pictures generation for checking the performance of Python representation
class ContinuousImageGenerator(Thread):
    """Threaded class for generation of continuous stream of noisy images and updating the ImageView widget."""

    meanGenTimes = np.zeros(200, dtype="uint")

    def __init__(self, imageWidget, refresh_delay_ms: int = 100, height: int = 100, width: int = 100, testPerformance: bool = False):
        Thread.__init__(self); self.height = height; self.width = width; self.testPerformance = testPerformance
        self.imageWidget = imageWidget; self.refresh_delay_ms = refresh_delay_ms

    def run(self):
        """
        Make continuous generation of noisy pictures and updating the ImageView widget from pyqtgraph for their showing.

        Returns
        -------
        None.

        """
        global flag_generation
        i = 0
        while(flag_generation):
            if self.testPerformance:
                t1 = time.time()
                if self.refresh_delay_ms == 0:  # if the delay between frames is 0, than the generation is unstable
                    self.refresh_delay_ms += 1  # make the delay at least 1 ms
            image = generate_noise_picture(self.height, self.width)
            self.imageWidget.setImage(image)
            time.sleep(self.refresh_delay_ms/1000)
            # If testing of Performance requested, then accumulating of passed times in the array performed
            if self.testPerformance:
                t2 = time.time()
                if i < np.size(self.meanGenTimes):
                    self.meanGenTimes[i] = np.uint(np.round((t2-t1)*1000, 0)); i += 1  # Add the passed time to the array
        # If generation stopped and the test of performance was asked, then print out the mean generation time
        if self.testPerformance:
            # Calculation of the final element that in the array is zero (passed time not saved)
            for j in range(np.size(self.meanGenTimes)):
                if self.meanGenTimes[j] == 0:
                    break
            self.meanGenTimes = self.meanGenTimes[0:j]  # Truckate array till the non-zerp element for mean value calculation
            mean_gen_t = np.uint(np.round(np.mean(self.meanGenTimes), 0))
            print("Mean generation time is:", mean_gen_t, "ms")


# %% Implementation of all windows inside the child class
class SimUscope(QMainWindow):
    """Create the GUI with buttons for testing image acquisition from the camera and making some image processing."""

    __flagGeneration = False  # Private class variable for recording state of continuous generation

    def __init__(self, img_height, img_width):
        super().__init__()
        self.imageGenerator = SingleImageGenerator(img_height, img_width); self.img_height = img_height; self.img_width = img_width
        self.img = np.zeros((self.img_height, self.img_width), dtype='uint8')  # Black initial image
        self.setWindowTitle("Simulation of uscope camera"); self.setGeometry(200, 200, 700, 600)
        self.imageWidget = pyqtgraph.ImageView(parent=self); self.imageWidget.ui.roiBtn.hide(); self.imageWidget.ui.menuBtn.hide()
        self.imageWidget.setImage(self.img)  # Set image for representation in the ImageView widget
        self.qwindow = QWidget()  # The composing of all buttons and frame for image representation into one main widget
        self.buttonGenSingleImg = QPushButton("Generate Single Pic"); self.buttonGenSingleImg.clicked.connect(self.generate_single_pic)
        self.buttonContinuousGen = QPushButton("Continuous Generation"); self.buttonContinuousGen.clicked.connect(self.generate_continuous_pics)
        self.buttonContinuousGen.setCheckable(True)
        grid = QGridLayout(self.qwindow); self.setLayout(grid)  # grid layout allows better layout of buttons and frames
        grid.addWidget(self.buttonGenSingleImg, 0, 0, 1, 1); grid.addWidget(self.buttonContinuousGen, 0, 1, 1, 1)
        grid.addWidget(self.imageWidget, 1, 0, 4, 4)  # the ImageView widget spans on 4 rows and 4 columns
        self.setCentralWidget(self.qwindow)  # Actually, allows to make both buttons and ImageView visible

    def generate_single_pic(self):
        """
        Handle clicking of Generate Single Picture. This method updates the image associated with ImageView widget.

        Returns
        -------
        None.

        """
        # !!!: Don't forget that waiting that thread complete its task guarantees all proper assignments (below)
        self.imageGenerator.start(); self.imageGenerator.join(); self.imageWidget.setImage(self.imageGenerator.image)
        self.imageGenerator = SingleImageGenerator(self.img_height, self.img_width)

    def generate_continuous_pics(self):
        """
        Handle clicking of Continuous Generation button. Generates continuously and updates the ImageView widget.

        Returns
        -------
        None.

        """
        self.__flagGeneration = not(self.__flagGeneration)  # changing the state of generation
        self.buttonContinuousGen.setDown(self.__flagGeneration)  # changing the visible state of button (clicked or not)
        global flag_generation
        flag_generation = self.__flagGeneration
        if (self.__flagGeneration):
            continuousImageGen = ContinuousImageGenerator(self.imageWidget, 1, self.img_height, self.img_width, True)
            continuousImageGen.start()


# %% Tests
if __name__ == "__main__":
    my_app = QApplication([])  # application without any command-line arguments
    my_app.setQuitOnLastWindowClosed(True)  # workaround for forcing the quit of the application window for returning to the kernel
    main_window = SimUscope(1000, 1000); main_window.show()

    my_app.exec()
