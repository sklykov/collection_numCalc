# -*- coding: utf-8 -*-
"""
Camera wrapper for imports and further utilization in OOP manner.

@author: ssklykov
"""
# %% Imports
from threading import Thread
from queue import Queue, Empty
import time
import pco


# %% Class wrapper
class PCOcamera(Thread):
    """Class for wrapping controls of the PCO camera and provided API features."""

    initialized = False  # Start the mail inifinite loop if the class initialized
    mainLoopTimeDelay = 25  # Internal constant - delaying for the main loop for receiving and process the commands

    def __init__(self, messagesQueue: Queue, exceptionsQueue: Queue, imageWidget):
        self.messagesQueue = messagesQueue  # For receiving the commands to stop / start live stream
        self.exceptionsQueue = exceptionsQueue  # For adding the exceptions that should stop the main program
        Thread.__init__(self)  # Initialize this class in the other thread
        self.initialized = True  # Additional flag for the start the loop in the run method
        self.imageWidget = imageWidget  # For updating the image on the pyqtgraph.ImageView
        # Initialization code for the camera
        try:
            self.cameraReference = pco.Camera()
            print("The PCO camera initialized")
        except ImportError:
            print("The PCO library is unavailable, check the installation. The camera not initialized!")
            self.cameraReference = None
        except ValueError:
            print("CAMERA NOT INITIALIZED! THE HANDLE TO IT - NONE")
            self.cameraReference = None
        self.max_width = 1024; self.max_height = 1024

    def run(self):
        """
        Keep the camera initializedand waiting the commands to start Live Stream / Single snap imaging.

        Returns
        -------
        None.

        """
        while self.initialized:
            # TODO: implement calling the functions for Live Stream, Getting single image, etc.

            # Checking for command of closing the camera
            if not(self.messagesQueue.empty()) and (self.messagesQueue.qsize() > 0):
                try:
                    message = self.messagesQueue.get_nowait()
                    if isinstance(message, str):
                        if message == "Close the camera" or message == "Stop" or message == "Stop Program":
                            try:
                                print("Received by the camera process:", message)
                                self.close()
                            except Exception as error:
                                print("Raised exception during closing the camera:", error)
                            finally:
                                self.initialized = False  # In any case stop the loop waiting the commands from the GUI
                        if message == "Stop Live Stream":
                            print("Camera stop live streaming")  # TODO
                        if message == "Start Live Stream":
                            print("Camera start live streaming")  # TODO
                        if message == "Snap single image":
                            self.snap_single_image()
                            print("Snap single image")  # TODO
                        if message == "Restore Full Frame":
                            print("Full frame image will be restored with sizes: ", self.max_width, self.max_height)  # TODO
                    if isinstance(message, tuple):
                        (command, parameters) = message
                        if command == "Crop Image":
                            print("Received sizes for cropping:", parameters)  # TODO
                except Empty:
                    pass

            time.sleep(self.mainLoopTimeDelay/1000)  # Delays of each step of processing of commands
        print("The threaded process for the camera is closed")

    def snap_single_image(self):
        """
        Get the single image from the camera.

        Returns
        -------
        None.

        """
        if self.cameraReference is not None:
            self.cameraReference.record(number_of_images=1, mode='sequence')
            self.cameraReference.wait_for_first_image()
            image, metadata = self.cameraReference.image()
            self.imageWidget.setImage(image)
            self.imageWidget.setPredefinedGradient('grey')

    def close(self):
        """
        Deinitialize the camera and close the connection to it.

        Returns
        -------
        None.

        """
        if self.cameraReference is not None:  # If the camera was initialized, then the cameraReference will be not None
            self.cameraReference.close()
        time.sleep(self.mainLoopTimeDelay/1000)
        print("The PCO camera closed")
