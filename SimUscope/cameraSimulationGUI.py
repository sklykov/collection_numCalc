# -*- coding: utf-8 -*-
"""
Simple GUI for representing of generated noisy image using PyQT.

@author: ssklykov
"""
# %% Imports
# import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QWidget, QGridLayout,
                             QSpinBox, QCheckBox, QVBoxLayout, QComboBox, QLabel)
from PyQt5.QtCore import Qt, QRect
import time
import numpy as np
import pyqtgraph
from cameraLiveStreamSimulation import SingleImageThreadedGenerator, ContinuousImageThreadedGenerator
from queue import Queue
import os
from skimage import io

# %% Some default values. The global value is omitted as the non-reliable for communication with the functions imported from modules
width_default = 1000; height_default = 1000  # Default width and height for generation of images


# %% Implementation of all windows inside the child class
class SimUscope(QMainWindow):
    """Create the GUI with buttons for testing image acquisition from the camera and making some image processing."""

    __flagGeneration = False  # Private class variable for recording state of continuous generation
    __flagTestPerformance = False  # Private class variable for switching between test state by using
    autoRange = True  # flag for handling user preference about possibility to zoom in/out to images

    def __init__(self, img_height, img_width):
        """Create overall UI inside the QMainWindow widget."""
        super().__init__()
        self.messages2Continuous_ImgGen = Queue(maxsize=4)  # Initialize message queue for communication with the Camera
        self.imageGenerator = SingleImageThreadedGenerator(img_height, img_width); self.img_height = img_height; self.img_width = img_width
        self.img = np.zeros((self.img_height, self.img_width), dtype='uint8')  # Black initial image
        self.setWindowTitle("Simulation of uscope camera"); self.setGeometry(200, 200, 800, 700)
        # PlotItem allows showing the axes and restrict the mouse usage over the image
        self.plot = pyqtgraph.PlotItem()
        self.plot.getViewBox().setMouseEnabled(False, False)  # !!!: Disable the possibility of move image by mouse
        # ImageView - for showing the generated image
        self.imageWidget = pyqtgraph.ImageView(view=self.plot)  # The main widget for image showing
        self.imageWidget.ui.roiBtn.hide(); self.imageWidget.ui.menuBtn.hide()   # Hide ROI, Norm buttons from the ImageView
        self.imageWidget.setImage(self.img)  # Set image for representation in the ImageView widget
        # QWidget - main widget window for grid layout
        self.qwindow = QWidget()  # The composing of all buttons and frame for image representation into one main widget
        # Selector of type of a camera - Simulated or PCO one
        self.cameraSelector = QComboBox(); self.cameraSelector.addItems(["Simulated Treaded", "Simulated Process", "PCO"])
        self.cameraSelLabel = QLabel("Camera Type")
        vboxSelector = QVBoxLayout(); vboxSelector.addWidget(self.cameraSelLabel); vboxSelector.addWidget(self.cameraSelector)
        self.cameraSelLabel.setAlignment(Qt.AlignCenter)  # Align the label text on the center
        # Push buttons for events evoking
        self.buttonGenSingleImg = QPushButton("Generate Single Pic"); self.buttonGenSingleImg.clicked.connect(self.generate_single_pic)
        self.buttonContinuousGen = QPushButton("Continuous Generation")  # Switches on/off continuous generation
        self.toggleTestPerformance = QCheckBox("Test Performance"); self.toggleTestPerformance.setEnabled(True)
        self.toggleTestPerformance.setChecked(False)  # setChecked - set the state of a button
        self.buttonContinuousGen.clicked.connect(self.generate_continuous_pics); self.buttonContinuousGen.setCheckable(True)
        self.exposureTime = QSpinBox(); self.exposureTime.setSingleStep(1); self.exposureTime.setSuffix(" ms")
        self.exposureTime.setPrefix("Exposure time: "); self.exposureTime.setMinimum(1); self.exposureTime.setMaximum(1000)
        self.exposureTime.setValue(100); self.exposureTime.adjustSize()
        self.switchMouseCtrlImage = QCheckBox("Handling Image by mouse"); self.switchMouseCtrlImage.setChecked(False)
        self.switchMouseCtrlImage.stateChanged.connect(self.activateMouseOnImage)
        self.saveSnapImg = QPushButton("Save Single Image"); self.saveSnapImg.clicked.connect(self.saveSingleSnapImg)
        self.saveSnapImg.setDisabled(True)  # disable the saving image before any image generated / displayed
        self.putROI = QPushButton("ROI selector"); self.putROI.clicked.connect(self.putROIonImage)
        # self.calculateImgFFT
        self.quitButton = QPushButton("Quit"); self.quitButton.setStyleSheet("color: red")
        self.quitButton.clicked.connect(self.quitClicked)
        # Grid layout below - the main layout pattern for all buttons and windos on the Main Window
        grid = QGridLayout(self.qwindow); self.setLayout(grid)  # grid layout allows better layout of buttons and frames
        grid.addLayout(vboxSelector, 0, 0, 1, 1)  # Add selector of a camera
        grid.addWidget(self.buttonGenSingleImg, 0, 1, 1, 1); grid.addWidget(self.buttonContinuousGen, 0, 2, 1, 1)
        grid.addWidget(self.toggleTestPerformance, 0, 3, 1, 1); grid.addWidget(self.exposureTime, 0, 4, 1, 1)
        # vbox below - container for Height / Width buttons
        vbox = QVBoxLayout(self.qwindow); self.widthButton = QSpinBox(); self.heightButton = QSpinBox(); vbox.addWidget(self.widthButton)
        self.heightButton.setPrefix("Height: "); self.widthButton.setPrefix("Width: "); vbox.addWidget(self.heightButton)
        grid.addLayout(vbox, 0, 5, 1, 1); self.widthButton.setSingleStep(2); self.heightButton.setSingleStep(2)
        self.widthButton.setMinimum(50); self.heightButton.setMinimum(50); self.widthButton.setMaximum(3000)
        self.heightButton.setMaximum(3000); self.widthButton.setValue(self.img_width); self.heightButton.setValue(self.img_height)
        grid.addWidget(self.quitButton, 0, 6, 1, 1); grid.addWidget(self.switchMouseCtrlImage, 7, 0, 1, 1)
        grid.addWidget(self.saveSnapImg, 7, 1, 1, 1); grid.addWidget(self.putROI, 7, 2, 1, 1)
        # Set valueChanged event handlers
        self.widthButton.valueChanged.connect(self.imgSizeChanged); self.heightButton.valueChanged.connect(self.imgSizeChanged)
        # ImageWidget should be central - for better representation of generated images
        grid.addWidget(self.imageWidget, 1, 0, 6, 6)  # the ImageView widget spans on ... rows and ... columns (2 values in the end)
        self.setCentralWidget(self.qwindow)  # Actually, allows to make both buttons and ImageView visible

    def generate_single_pic(self):
        """
        Handle clicking of Generate Single Picture. This method updates the image associated with ImageView widget.

        Returns
        -------
        None.

        """
        # !!!: Don't forget that waiting that thread complete its task guarantees all proper assignments (below)
        self.imageGenerator.start(); self.imageGenerator.join()
        self.imageWidget.setImage(self.imageGenerator.image, autoRange=self.autoRange)
        self.imageGenerator = SingleImageThreadedGenerator(self.img_height, self.img_width)
        if not(self.saveSnapImg.isEnabled()):   # activate saving images, since some image generated and displayed
            self.saveSnapImg.setEnabled(True)

    def generate_continuous_pics(self):
        """
        Handle clicking of Continuous Generation button. Generates continuously and updates the ImageView widget.

        Returns
        -------
        None.

        """
        self.__flagGeneration = not(self.__flagGeneration)  # changing the state of generation
        self.buttonContinuousGen.setDown(self.__flagGeneration)  # changing the visible state of button (clicked or not)
        if (self.__flagGeneration):
            self.toggleTestPerformance.setDisabled(True)  # Disable the check box for preventing test on during continuous generation
            self.exposureTime.setDisabled(True)  # Disable the exposure time control
            self.widthButton.setDisabled(True); self.heightButton.setDisabled(True)
            # Below - initialization of the imported class for continous image generation
            self.continuousImageGen = ContinuousImageThreadedGenerator(self.imageWidget, self.messages2Continuous_ImgGen,
                                                                       self.exposureTime.value(), self.img_height, self.img_width,
                                                                       self.toggleTestPerformance.isChecked(), self.autoRange)
            self.continuousImageGen.start()  # Run the threaded code
            if not(self.saveSnapImg.isEnabled()):   # activate saving images, since some image generated and displayed
                time.sleep(self.exposureTime.value()/1000)  # wait at least 1 exposure time before activate the saving button
                self.saveSnapImg.setEnabled(True)
        else:
            if not(self.messages2Continuous_ImgGen.full()):
                self.messages2Continuous_ImgGen.put_nowait("Stop Generation")  # Send the message to stop continuous generation
            if (self.continuousImageGen.is_alive()):  # if the threaded process is still running
                self.continuousImageGen.join()  # wait the stop of threaded process ending
            # Returns buttons to active state below
            self.toggleTestPerformance.setEnabled(True); self.exposureTime.setEnabled(True)
            self.widthButton.setEnabled(True); self.heightButton.setEnabled(True)

    def closeEvent(self, closeEvent):
        """
        Rewrites the default behaviour of clicking on the X button on the main window GUI.

        Parameters
        ----------
        closeEvent : QWidget Close Event
            Needed by the API.

        Returns
        -------
        None.

        """
        # Below - sending message to stop continuous generation
        if not(self.messages2Continuous_ImgGen is None) and not(self.messages2Continuous_ImgGen.full()) and self.__flagGeneration:
            self.messages2Continuous_ImgGen.put_nowait("Stop Generation")  # Send the message to stop continuous generation
            if (self.continuousImageGen.is_alive()):  # if the threaded generation process is still running
                self.continuousImageGen.join()  # wait the stop of threaded process ending
            closeEvent.accept()  # Maybe redundant, but this is explicit accepting quit event (could be refused if asked for example)

    def quitClicked(self):
        """
        Handle the clicking event on the Quit Button.

        Sets the global variables to False state. Waits for threads stop running. Quits the Main window.

        Returns
        -------
        None.

        """
        if not(self.messages2Continuous_ImgGen is None) and not(self.messages2Continuous_ImgGen.full()) and self.__flagGeneration:
            self.messages2Continuous_ImgGen.put_nowait("Stop Generation")  # Send the message to stop continuous generation
            if (self.continuousImageGen.is_alive()):  # if the threaded generation process is still running
                self.continuousImageGen.join()  # wait the stop of threaded process ending
        self.close()  # Calls the closing event for QMainWindow

    def imgSizeChanged(self):
        """
        Handle changing of image width or height. Allows to pick up values for single image generation and continuous one.

        Returns
        -------
        None.

        """
        self.img_width = self.widthButton.value(); self.img_height = self.heightButton.value()
        self.imageGenerator = SingleImageThreadedGenerator(self.img_height, self.img_width)

    def activateMouseOnImage(self):
        """
        Activate or Deactivate mouse handling of images - zooming in / out.

        Returns
        -------
        None.

        """
        self.plot.getViewBox().setMouseEnabled(self.switchMouseCtrlImage.isChecked(), self.switchMouseCtrlImage.isChecked())
        if not(self.switchMouseCtrlImage.isChecked()):  # switched from enabled to disabled state
            self.autoRange = True  # All new images will be auto-centralized
            self.plot.getViewBox().enableAutoRange(enable=True)  # "Centralize" the image by auto range on axes
        else:
            self.autoRange = False

    def saveSingleSnapImg(self):
        """
        Save the current displayed image (but it isn't thread-save!).

        Returns
        -------
        None.

        """
        snap_img = self.imageWidget.getImageItem().image  # get the current displayed image
        composedName = "SnapImage.tiff"; composedPath = os.path.join(os.getcwd(), composedName)  # getting some default image name
        io.imsave(composedPath, snap_img, plugin='tifffile')  # save tiff file by using embedded tifffile plugin

    def putROIonImage(self):
        """
        Put ROI on the displayed image.

        Returns
        -------
        None.

        """
        qrect = QRect(0, 0, self.img_height, self.img_width)  # Bounding to image rectangle
        if hasattr(self, "roi"):  # Checking of the class has the attribute already "roi"
            self.removeROI()  # Cleaning existed roi
        self.roi = pyqtgraph.ROI((self.img_height//2 - 50, self.img_width//2 - 50), size=(100, 100),
                                 rotatable=False, removable=True, maxBounds=qrect)  # Create the ROI object
        self.plot.addItem(self.roi)  # Add ROI object on the image
        self.roi.sigRemoveRequested.connect(self.removeROI)  # Register handling of ROI tool

    def removeROI(self):
        """
        Remove created ROI from the displayed image.

        Returns
        -------
        None.

        """
        self.plot.removeItem(self.roi)  # Remove added roi


# %% Tests
if __name__ == "__main__":
    my_app = QApplication([])  # application without any command-line arguments
    my_app.setQuitOnLastWindowClosed(True)  # workaround for forcing the quit of the application window for returning to the kernel
    main_window = SimUscope(width_default, height_default); main_window.show()
    my_app.exec()  # Exit of the main program
