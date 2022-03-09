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
from checkExceptionsInLive import CheckMessagesForExceptions
from cameraCtrl import PCOcamera

# %% Some default values. The global value is omitted as the non-reliable for communication with the functions imported from modules
width_default = 1000; height_default = 1000  # Default width and height for generation of images


# %% Implementation of all windows inside the child class
class SimUscope(QMainWindow):
    """Create the GUI with buttons for testing image acquisition from the camera and making some image processing."""

    __flagGeneration = False  # Private class variable for recording state of continuous generation
    __flagTestPerformance = False  # Private class variable for switching between test state by using
    autoRange = True  # flag for handling user preference about possibility to zoom in/out to images

    def __init__(self, img_height, img_width, applicationHandle: QApplication):
        """Create overall UI inside the QMainWindow widget."""
        super().__init__()
        self.applicationHandle = applicationHandle  # handle to the main application for exit it in the appropriate place
        self.messages2Camera = Queue(maxsize=4)  # Initialize message queue for communication with the camera (simulated or not)
        self.exceptionsQueue = Queue(maxsize=4)  # Initialize separate queue for spreading and handling Exceptions occured within modules
        self.imageGenerator = SingleImageThreadedGenerator(img_height, img_width); self.img_height = img_height; self.img_width = img_width
        self.img = np.zeros((self.img_height, self.img_width), dtype='uint8')  # Black initial image
        self.setWindowTitle("Simulation of uscope camera"); self.setGeometry(200, 200, 840, 800)
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
        self.cameraSelector = QComboBox(); self.cameraSelector.addItems(["Simulated Threaded", "PCO"])
        self.cameraSelector.currentTextChanged.connect(self.activeCameraChanged)  # Attach handlers for camera choosing
        self.cameraSelLabel = QLabel("Camera Type"); self.cameraSelector.setEditable(True)  # setEditable is needed for setAlignment
        self.cameraSelector.lineEdit().setAlignment(Qt.AlignCenter); self.cameraSelector.lineEdit().setReadOnly(True)
        vboxSelector = QVBoxLayout(); vboxSelector.addWidget(self.cameraSelLabel); vboxSelector.addWidget(self.cameraSelector)
        self.cameraSelLabel.setAlignment(Qt.AlignCenter)  # Align the label text on the center
        # ROI size selectors
        self.heightROI = QSpinBox(); self.heightROI.setSingleStep(2); self.heightROI.setMaximum(self.img_width)  # swapped
        self.widthROI = QSpinBox(); self.widthROI.setSingleStep(2); self.widthROI.setMaximum(self.img_height)  # swapped
        self.heightROI.setMinimum(2); self.widthROI.setMinimum(2); self.heightROI.setValue(100); self.widthROI.setValue(100)
        self.widthROI.setAlignment(Qt.AlignCenter); self.heightROI.setAlignment(Qt.AlignCenter)
        self.widthROI.setPrefix("ROI width: "); self.heightROI.setPrefix("ROI height: ")
        vboxROI = QVBoxLayout(); vboxROI.addWidget(self.widthROI); vboxROI.addWidget(self.heightROI)
        self.widthROI.valueChanged.connect(self.roiSizesSpecified); self.heightROI.valueChanged.connect(self.roiSizesSpecified)
        # Push buttons for events evoking
        self.buttonGenSingleImg = QPushButton("Generate Single Pic"); self.buttonGenSingleImg.clicked.connect(self.generate_single_pic)
        self.buttonContinuousGen = QPushButton("Continuous Generation")  # Switches on/off continuous generation
        self.toggleTestPerformance = QCheckBox("Test Performance"); self.toggleTestPerformance.setEnabled(True)
        self.toggleTestPerformance.setChecked(False)  # setChecked - set the state of a button
        self.buttonContinuousGen.clicked.connect(self.generate_continuous_pics); self.buttonContinuousGen.setCheckable(True)
        self.exposureTime = QSpinBox(); self.exposureTime.setSingleStep(1); self.exposureTime.setSuffix(" ms")
        self.exposureTime.setPrefix("Exposure time: "); self.exposureTime.setMinimum(1); self.exposureTime.setMaximum(1000)
        self.exposureTime.setValue(100); self.exposureTime.adjustSize(); self.exposureTime.valueChanged.connect(self.exposureTimeChanged)
        self.switchMouseCtrlImage = QCheckBox("Handling Image by mouse"); self.switchMouseCtrlImage.setChecked(False)
        self.switchMouseCtrlImage.stateChanged.connect(self.activateMouseOnImage)
        self.saveSnapImg = QPushButton("Save Single Image"); self.saveSnapImg.clicked.connect(self.saveSingleSnapImg)
        self.saveSnapImg.setDisabled(True)  # disable the saving image before any image generated / displayed
        self.putROI = QPushButton("ROI selector"); self.putROI.clicked.connect(self.putROIonImage)
        self.generateException = QPushButton("Generate Exception"); self.generateException.clicked.connect(self.genMessageWithException)
        # self.calculateImgFFT
        self.quitButton = QPushButton("Quit"); self.quitButton.setStyleSheet("color: red")
        self.quitButton.clicked.connect(self.quitClicked)
        # Grid layout below - the main layout pattern for all buttons and windows put on the Main Window (GUI)
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
        grid.addLayout(vboxROI, 7, 3, 1, 1); grid.addWidget(self.generateException, 7, 6, 1, 1)
        # Set valueChanged event handlers
        self.widthButton.valueChanged.connect(self.imgSizeChanged); self.heightButton.valueChanged.connect(self.imgSizeChanged)
        # ImageWidget should be central - for better representation of generated images
        grid.addWidget(self.imageWidget, 1, 0, 6, 6)  # the ImageView widget spans on ... rows and ... columns (2 values in the end)
        self.setCentralWidget(self.qwindow)  # Actually, allows to make both buttons and ImageView visible
        # Initiliaze and start the Exception checker and associate it with the initialized Quit button
        self.exceptionChecker = CheckMessagesForExceptions(self.exceptionsQueue, self.quitButton, period_checks_ms=25)
        self.exceptionChecker.start()  # Start the Exception checker

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
            # Below - initialization of the imported class for continuous image generation
            if self.cameraSelector.currentText() == "Simulated Threaded":
                self.continuousImageGen = ContinuousImageThreadedGenerator(self.imageWidget, self.messages2Camera,
                                                                           self.exposureTime.value(), self.img_height, self.img_width,
                                                                           self.toggleTestPerformance.isChecked(), self.autoRange)
                self.continuousImageGen.start()  # Run the threaded code for continuous updating of showing image
            elif self.cameraSelector.currentText() == "PCO":
                self.cameraHandle = PCOcamera(self.exceptionsQueue)  # Initialize the PCO camera

            if not(self.saveSnapImg.isEnabled()):   # activate saving single image button, since some image generated and displayed
                time.sleep(self.exposureTime.value()/1000)  # wait at least 1 exposure time before activate the saving button
                self.saveSnapImg.setEnabled(True)
        else:
            if not(self.messages2Camera.full()) and (self.cameraSelector.currentText() == "Simulated Threaded"):
                self.messages2Camera.put_nowait("Stop Generation")  # Send the message to stop continuous generation
            if hasattr(self, "continuousImageGen"):
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
        if not(self.messages2Camera is None) and not(self.messages2Camera.full()) and self.__flagGeneration:
            self.messages2Camera.put_nowait("Stop Generation")  # Send the message to stop continuous generation
            if hasattr(self, "continuousImageGen"):
                if (self.continuousImageGen.is_alive()):  # if the threaded generation process is still running
                    self.continuousImageGen.join()  # wait the stop of threaded process ending
        if self.exceptionChecker.is_alive():
            self.exceptionsQueue.put_nowait("Stop Exception Checker")
            self.exceptionChecker.join()
        closeEvent.accept()  # Maybe redundant, but this is explicit accepting quit event (could be refused if asked for example)
        self.applicationHandle.exit()  # Exit the main application, returning to the calling it kernel

    def quitClicked(self):
        """
        Handle the clicking event on the Quit Button.

        Sets the global variables to False state. Waits for threads stop running. Quits the Main window.

        Returns
        -------
        None.

        """
        if not(self.messages2Camera is None) and not(self.messages2Camera.full()) and self.__flagGeneration:
            self.messages2Camera.put_nowait("Stop Generation")  # Send the message to stop continuous generation
            if hasattr(self, "continuousImageGen"):
                if (self.continuousImageGen.is_alive()):  # if the threaded generation process is still running
                    self.continuousImageGen.join()  # wait the stop of threaded process ending
        if self.exceptionChecker.is_alive():
            self.exceptionsQueue.put_nowait("Stop Exception Checker")
            self.exceptionChecker.join()
        self.close()  # Calls the closing event for QMainWindow

    def imgSizeChanged(self):
        """
        Handle changing of image width or height. Allows to pick up values for single image generation and continuous one.

        Returns
        -------
        None.

        """
        self.img_width = self.widthButton.value(); self.img_height = self.heightButton.value()
        self.imageGenerator = SingleImageThreadedGenerator(self.img_height, self.img_width)  # reinitialize single generator

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
        qrect = QRect(2, 2, self.img_height, self.img_width)  # Bounding of ROI to the image (ROI not movable outside of an image)
        if hasattr(self, "roi"):  # Checking of the class has the attribute already "roi" => some ROI already drawn
            self.removeROI()  # Cleaning existed (drawn) roi from the image for refreshing it
        # ROI will be put in the middle of an image, assuming it has size more than 100 pixels, width and height swapped again
        # Create the ROI object that is non rotatable, removable and expanding only evenly (snapSize and scaleSnap)
        self.roi = pyqtgraph.ROI((self.img_width//2 - 50, self.img_height//2 - 50), size=(100, 100), snapSize=2.0,
                                 scaleSnap=True, rotatable=False, removable=True, maxBounds=qrect)
        self.plot.addItem(self.roi)  # Add ROI object on the image
        self.roi.sigRemoveRequested.connect(self.removeROI)  # Register handling of removing of ROI
        self.roi.sigRegionChangeFinished.connect(self.roiSizeChanged)

    def removeROI(self):
        """
        Remove created ROI from the displayed image.

        Returns
        -------
        None.

        """
        self.plot.removeItem(self.roi)  # Remove added roi

    def roiSizesSpecified(self):
        """
        Adjust input value for preventing odd values specification within the QSpinBoxes.

        Returns
        -------
        None.

        """
        if self.heightROI.value() % 2 != 0:
            self.heightROI.setValue(self.heightROI.value()+1)
        if self.widthROI.value() % 2 != 0:
            self.widthROI.setValue(self.widthROI.value()+1)
        if hasattr(self, "roi"):
            self.roi.setSize((self.widthROI.value(), self.heightROI.value()))

    def roiSizeChanged(self):
        """
        Transfer changed by the user sizes of ROI on the image to the buttons representing them on GUI.

        Returns
        -------
        None.

        """
        (w, h) = self.roi.size(); w = int(w); h = int(h)
        self.widthROI.setValue(w); self.heightROI.setValue(h)

    def genMessageWithException(self):
        """
        Put the exception into the Queue used for communication with the camera for testing quit/stop handling.

        Returns
        -------
        None.

        """
        if not(self.exceptionsQueue.full()):
            self.exceptionsQueue.put_nowait(Exception("Generated Test Exception"))

    def exposureTimeChanged(self):
        """
        Handle changing of exposure time by the user.

        Returns
        -------
        None.

        """
        pass

    def activeCameraChanged(self):
        """
        Handle changing of active (selected) camera.

        Returns
        -------
        None.

        """
        if self.cameraSelector.currentText() == "PCO":
            self.widthButton.setDisabled(True); self.heightButton.setDisabled(True)
            self.generateException.setVisible(False)  # Remove button for testing of handling of generated Exceptions
            # Changing the titles of the buttons for controlling getting the images (from the camera or generated ones)
            self.buttonGenSingleImg.setText("Single Snap Image"); self.buttonContinuousGen.setText("Live Stream")
        elif self.cameraSelector.currentText() == "Simulated Threaded":
            self.widthButton.setEnabled(True); self.heightButton.setEnabled(True)
            if not(self.generateException.isVisible()):  # return the visibility of the button
                self.generateException.setVisible(True)
            self.buttonGenSingleImg.setText("Generate Single Pic"); self.buttonContinuousGen.setText("Continuous Generation")


# %% Tests
if __name__ == "__main__":
    my_app = QApplication([])  # application without any command-line arguments
    # my_app.setQuitOnLastWindowClosed(True)  # workaround for forcing the quit of the application window for returning to the kernel
    main_window = SimUscope(width_default, height_default, my_app); main_window.show()
    my_app.exec()  # Execute the application in the main kernel
