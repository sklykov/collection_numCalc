# -*- coding: utf-8 -*-
"""
Class wrapper for controlling PCO or simulated camera using separate process.

It implements API functions provided in the pco controlling library https://pypi.org/project/pco/
For dependency list - see the imports (the simplest form).
@author: ssklykov
"""
# %% Imports
from multiprocessing import Process, Queue
from queue import Empty, Full
import time
import pco
import numpy as np


# %% Class wrapper
class CameraWrapper(Process):
    """Class for wrapping controls of the PCO camera and provided API features."""

    initialized = False  # Start the mail inifinite loop if the class initialized
    mainLoopTimeDelay = 25  # Internal constant - delaying for the main loop for receiving and process the commands
    liveStream: bool  # force type checking
    exposure_time_ms: int
    camera_type: str
    max_width: int; max_height: int
    image_width: int; image_height: int
    cameraReference = None

    def __init__(self, messagesQueue: Queue, exceptionsQueue: Queue, imagesQueue: Queue, messagesToCaller: Queue,
                 exposure_time_ms: int, img_width: int, img_height: int, camera_type: str = "Simulated"):
        Process.__init__(self)  # Initialize this class on the separate process with it's own memory and core
        self.messagesQueue = messagesQueue  # For receiving the commands to stop / start live stream
        self.exceptionsQueue = exceptionsQueue  # For adding the exceptions that should stop the main program
        self.messagesToCaller = messagesToCaller  # For sending internal messages from this class for debugging
        self.imagesQueue = imagesQueue  # For storing the acquired images
        self.liveStream = False  # Set default live stream state to false
        self.exposure_time_ms = exposure_time_ms  # Initializing with the default exposure time
        self.camera_type = camera_type  # Type of initialized camera
        # Initialization code for the PCO camera -> MOVED to the run() because it's impossible to pickle handle to the camera
        if self.camera_type == "PCO":
            # All the camera initialization code moved to the run() method!
            # It's because the camera handle returned by the DLL library is not pickleable by the pickle method !!!
            self.initialized = True  # Additional flag for the start the loop in the run method (run Process!)
            self.max_width = 2048; self.max_height = 2048
        # Initialization code for the Simulated camera
        elif self.camera_type == "Simulated":
            self.cameraReference = None  # no associated library call to any API functions
            self.initialized = True  # Additional flag for the start the loop in the run method
            self.max_width = img_width; self.max_height = img_height
            self.image_height = img_height; self.image_width = img_width
            try:
                self.generate_noise_picture()
                self.messagesToCaller.put_nowait("The Simulated camera initialized")
            except Exception as e:
                # Only arises if image width or height are too small (less than 2 pixels)
                self.messagesQueue.put_nowait(str(e))
                self.exceptionsQueue.put_nowait(e)
                self.initialized = False
        # Stop initialization because the camera type could be recognized
        else:
            self.cameraReference = None
            self.messagesToCaller.put_nowait("The specified type of the camera hasn't been implemented")
            self.initialized = False  # Additional flag for the start the loop in the run method
            self.exceptionsQueue.put_nowait(Exception("The specified type of the camera not implemented"))

    def run(self):
        """
        Keep the camera initializedand waiting the commands to start Live Stream / Single snap imaging.

        Returns
        -------
        None.

        """
        # !!! Below - initialization code for the PCO camera, because in the __init__() method it's impossible to make,
        # because the handle returned by the call pco.Camera() returns the unpickleable object
        if self.camera_type == "PCO":
            try:
                self.cameraReference = pco.Camera()  # open connection to the camera
                # printing statement won't be redirected to stdout, thus sending this message to the queue
                self.messagesToCaller.put_nowait("The PCO camera initialized")
                self.cameraReference.set_exposure_time(self.exposure_time_ms/1000)
                self.messagesToCaller.put_nowait("Default exposure time is set for the camera")
                self.messagesToCaller.put_nowait("The camera status report: " + str(self.cameraReference.sdk.get_camera_health_status()))
                self.initialized = True  # Additional flag for the start the loop in the run method
                self.imagesQueue.put_nowait("The PCO camera initialized")
            except ImportError as import_err:
                self.messagesToCaller.put_nowait("The PCO library is unavailable, check the installation. The camera not initialized!")
                self.cameraReference = None
                self.exceptionsQueue.put_nowait(import_err)
            except ValueError:
                self.messagesToCaller.put_nowait("CAMERA NOT INITIALIZED! THE HANDLE TO IT - 'NONE'")
                self.imagesQueue.put_nowait("The PCO camera initialized")
                self.cameraReference = None
        self.messagesToCaller.put_nowait(self.camera_type + " camera Process has been launched")  # Send the command for debugging
        # The main loop - the handler in the Process loop
        while self.initialized:
            # Checking for commands created by clicking buttons or by any events
            if not(self.messagesQueue.empty()) and (self.messagesQueue.qsize() > 0):
                try:
                    message = self.messagesQueue.get_nowait()  # get the message from the main controlling GUI
                    if isinstance(message, str):
                        # Close the Camera
                        if message == "Close the camera" or message == "Stop" or message == "Stop Program" or message == "Close Camera":
                            try:
                                self.messagesToCaller.put_nowait("Received by the Camera: " + message)
                                self.close()  # closing the connection to the camera and release all resources
                            except Exception as error:
                                self.messagesToCaller.put_nowait("Raised exception during closing the camera:", error)
                                self.exceptionsQueue.put_nowait(error)  # re-throw to the main program the error
                            finally:
                                self.initialized = False; break  # In any case stop the loop waiting the commands from the GUI
                        # Live stream mode
                        if message == "Start Live Stream":
                            self.messagesToCaller.put_nowait("Camera start live streaming")
                            try:
                                self.live_imaging()  # call the function
                            except Exception as error:
                                self.messagesToCaller.put_nowait("Error string: " + str(error))
                                self.exceptionsQueue.put_nowait(error)
                        # Acquiring single image
                        if message == "Snap single image":
                            try:
                                # The single acquired image is sent back to the calling controlling program via Queue
                                self.snap_single_image()
                                self.messagesToCaller.put_nowait("Single image snap performed")
                            except Exception as e:
                                # Any encountered exceptions should be reported to the main controlling program
                                self.close()  # An attempt to close the camera
                                self.initialized = False  # Stop this running loop
                                self.exceptionsQueue.put_nowait(e)  # Send to the main controlling program the caught Exception e
                        # Restore full frame
                        if message == "Restore Full Frame":
                            self.messagesToCaller.put_nowait(("Full frame restored: " + str((self.max_width, self.max_height))))
                            self.restoreFullFrame()  # TODO
                    if isinstance(message, tuple):
                        (command, parameters) = message
                        if command == "Crop Image":
                            # Send back for debugging crop parameters - below
                            self.messagesToCaller.put_nowait("Crop coordinates: " + str(parameters))
                            (yLeftUpper, xLeftUpper, height, width) = parameters
                            self.cropImage(yLeftUpper, xLeftUpper, width, height)  # TODO
                        # Set exposure time
                        if command == "Set exposure time":
                            self.setExposureTime(parameters)
                        # Set the new image sizes for Simulated camera
                        if command == "Change simulate picture sizes to:":
                            (command, parameters) = message
                            (width, height) = parameters
                            self.updateSimulatedSizes(width, height)
                    # Exceptions handling => close the camera if it receives from other parts of the program the exception
                    if isinstance(message, Exception):
                        print("Camera will be stopped because of throw from the main GUI exception")
                        self.close()
                except Empty:
                    pass

            time.sleep(self.mainLoopTimeDelay/1000)  # Delays of each step of processing of commands
        self.messagesToCaller.put_nowait("run() of Process finished for " + self.camera_type + " camera")  # DEBUG

    def snap_single_image(self):
        """
        Get the single image from the camera.

        Returns
        -------
        None.

        """
        if (self.cameraReference is not None) and (self.camera_type == "PCO"):
            self.cameraReference.record(number_of_images=1, mode='sequence')  # setup the camera to acquire a single image
            self.cameraReference.wait_for_first_image()
            image, metadata = self.cameraReference.image()  # get the single image
            self.imagesQueue.put_nowait(image)  # put the image to the queue for getting it in the main thread
        elif (self.cameraReference is None) and (self.camera_type == "PCO"):
            self.imagesQueue.put_nowait("String replacer of an image")
        elif self.camera_type == "Simulated":
            image = self.generate_noise_picture()  # No need to evoke try, because the width and height conformity already checked
            self.imagesQueue.put_nowait(image)

    def live_imaging(self):
        """
        Make of Live Imaging stream for the PCO camera.

        Returns
        -------
        None.

        """
        self.liveStream = True  # flag for the infinite loop for the streaming images continuously
        # General handle for live streaming - starting with acquiring of the first image
        while (self.liveStream):
            # then it's supposed that the camera has been properly initialized - below
            if (self.camera_type == "PCO") and (self.cameraReference is not None):
                self.cameraReference.record(number_of_images=25, mode='ring buffer')  # configure the live stream acquisition
                # make the loop below for infinite live stream, that could be stopped only by receiving the command or exception
                self.cameraReference.wait_for_first_image()  # wait that the image acquired actually
                image, metadata = self.cameraReference.image()  # get the acquired image
                if not(self.imagesQueue.full()):
                    try:
                        self.imagesQueue.put_nowait(image)  # put the image to the queue for getting it in the main thread
                    except Full:
                        pass  # do nothing for now about the overloaded queue
            elif (self.camera_type == "PCO") and (self.cameraReference is None):
                # Substituion of actual image generation by the PCO camera
                time.sleep(self.exposure_time_ms/1000)
                self.imagesQueue.put_nowait("Live Image substituted by this string")
            elif self.camera_type == "Simulated":
                image = self.generate_noise_picture()  # simulate some noise image
                if not(self.imagesQueue.full()):
                    self.imagesQueue.put_nowait(image)
                time.sleep(self.exposure_time_ms/1000)  # Delay due to the simulated exposure
            # Below - checking for the command "Stop Live stream
            if not(self.messagesQueue.empty()) and (self.messagesQueue.qsize() > 0):
                try:
                    message = self.messagesQueue.get_nowait()  # get the message from the main controlling GUI
                    if isinstance(message, str):
                        if message == "Stop Live Stream":
                            self.messagesToCaller.put_nowait("Camera stop live streaming")
                            if (self.camera_type == "PCO") and (self.cameraReference is not None):
                                self.cameraReference.stop()  # stop the live stream from the PCO  camera
                            self.imagesQueue.empty()  # delete all images from the queue for stopping their representation
                            self.liveStream = False; break
                    elif isinstance(message, Exception):
                        self.messagesToCaller.put_nowait("Camera stop live streaming because of the reported error")
                        if (self.camera_type == "PCO") and (self.cameraReference is not None):
                            self.cameraReference.stop()  # stop the live stream from the PCO camera
                        self.messagesQueue.put_nowait(message)  # setting for run() method again the error report for stopping the camera
                        self.liveStream = False; break
                except Empty:
                    pass

    def setExposureTime(self, exposure_time_ms: int):
        """
        Set exposure time for the camera.

        Parameters
        ----------
        exposure_time_ms : int
            Provided value for exposure time from GUI.

        Returns
        -------
        None.

        """
        if isinstance(exposure_time_ms, int):
            if exposure_time_ms == 0:  # exposure time cannot be 0
                exposure_time_ms = 1
            self.exposure_time_ms = exposure_time_ms
            if self.cameraReference is not None:  # if the camera is really activated, then call the function
                self.cameraReference.set_exposure_time(self.exposure_time_ms/1000)
            print("The set exposure time:", self.exposure_time_ms)

    def cropImage(self, yLeftUpper: int, xLeftUpper: int, width: int, height: int):
        """
        Crop selected ROI from the image.

        Parameters
        ----------
        yLeftUpper : int
            y coordinate of left upper corner of ROI region.
        xLeftUpper : int
            x coordinate of left upper corner of ROI region.
        width : int
            ROI width.
        height : int
            ROI height.

        Returns
        -------
        None.

        """
        # For simulated camera only reassign the image height and width
        if self.camera_type == "Simulated":
            self.image_width = width; self.image_height = height

    def restoreFullFrame(self):
        """
        Restore full size image.

        Returns
        -------
        None.

        """
        # For simulated camera only reassign the image height and width
        if self.camera_type == "Simulated":
            self.image_width = self.max_width; self.image_height = self.max_height

    def updateSimulatedSizes(self, width: int, height: int):
        """
        Update of sizes of a simulated image.

        Parameters
        ----------
        width : int
            Updated image width.
        height : int
            Updated image height.

        Returns
        -------
        None.

        """
        self.image_width = width; self.max_width = width; self.image_height = height; self.max_height = height

    def close(self):
        """
        Deinitialize the camera and close the connection to it.

        Returns
        -------
        None.

        """
        #  If the camera - PCO, then call close() function from the pco module
        if (self.camera_type == "PCO") and (self.cameraReference is not None):
            self.cameraReference.close()
        time.sleep(self.mainLoopTimeDelay/1000)  #
        self.messagesToCaller.put_nowait("The " + self.camera_type + " camera closed")

    def generate_noise_picture(self, pixel_type: str = 'uint8') -> np.ndarray:
        """
        Generate of a noise image with even distribution of noise (pixel values) on that.

        Parameters
        ----------
        height : int
            Height of a generated image.
        width : int
            Width of a generated image.
        pixel_type : str, optional
            Type of pixels in an image. The default is 'uint8'.

        Raises
        ------
        Exception
            When the specified height or width are less than 2.

        Returns
        -------
        img : np.ndarray
            Generate image with even noise.

        """
        img = np.zeros((1, 1), dtype='uint8')
        height = self.image_height; width = self.image_width
        if (height >= 2) and (width >= 2):
            if pixel_type == 'uint8':
                img = np.random.randint(0, high=255, size=(height, width), dtype='uint8')
            if pixel_type == 'float':
                img = np.random.rand(height, width)
        else:
            raise Exception("Specified height or width are less than 2")

        return img


# %% It's testing and unnecessary code, only valuable to check if the real camera is initialized before running the GUI
if __name__ == "__main__":
    messagesQueue = Queue(maxsize=5); exceptionsQueue = Queue(maxsize=2); imagesQueue = Queue(maxsize=2)
    messagesToCaller = Queue(maxsize=5); exposure_time_ms = 100; img_width = 100; img_height = 100
    camera = CameraWrapper(messagesQueue, exceptionsQueue, imagesQueue, messagesToCaller,
                           exposure_time_ms, img_width, img_height, camera_type="PCO")
    camera.start(); time.sleep(8)
    if not(messagesToCaller.empty()):
        while not(messagesToCaller.empty()):
            try:
                print(messagesToCaller.get_nowait())
            except Empty:
                pass
    messagesQueue.put_nowait("Stop"); time.sleep(5)
    if not(messagesToCaller.empty()):
        while not(messagesToCaller.empty()):
            try:
                print(messagesToCaller.get_nowait())
            except Empty:
                pass
    camera.join()
