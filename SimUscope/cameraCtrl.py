# -*- coding: utf-8 -*-
"""
Camera wrapper for imports and further utilization in OOP manner.

@author: ssklykov
"""
# %% Imports
from threading import Thread
from queue import Queue, Empty
import time


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
        # TODO: initialization code for the camera
        print("The PCO camera initialized")

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
                            except Exception:
                                pass
                            finally:
                                self.initialized = False  # In any case stop the loop
                        if message == "Stop Live Stream":
                            print("Camera stop live streaming")  # TODO
                        if message == "Start Live Stream":
                            print("Camera start live streaming")  # TODO
                        if message == "Snap single image":
                            print("Snap single image")  # TODO
                except Empty:
                    pass

            time.sleep(self.mainLoopTimeDelay/1000)  # Delays of each step of processing of commands

    def close(self):
        """
        Deinitialize the camera and close the connection.

        Returns
        -------
        None.

        """
        # TODO: camera deinitialize
        time.sleep(self.mainLoopTimeDelay/1000)
        print("The PCO camera closed")
