# -*- coding: utf-8 -*-
"""
Attempt to access the built-in camera in the laptop.

@author: sklykov

"""

# %% Global Imports
import matplotlib.pyplot as plt
import cv2

# %% References
# 1) https://docs.opencv.org/4.10.0/dd/d43/tutorial_py_video_display.html
# 2) https://pyimagesearch.com/2014/11/03/display-matplotlib-rgb-image/  (conversion of BGR opencv image to RGB)

# %% Tests
if __name__ == "__main__":
    # Loop for finding the available index of camera if any is available
    camera_handle = None; plt.close('all')

    # Iteration over some range of indices for searching of an available camera
    for i in range(0, 6, 1):

        # Not necessary to surround the initialization call with try except structure, the exception will be prinout only
        # try:
        #     camera = cv2.VideoCapture(i)
        #     try:
        #         camera_is_available = camera.isOpened()
        #         if camera_is_available:
        #             print(f"Camera with index {i} is available and will be used")
        #             camera_handle = camera; break
        #     except Exception:
        #         print(f"Camera with index {i} is not available")
        # except Exception:
        #     print(f"Camera with index {i} is not available")

        # Simple iterating over all indices hard-coded above and checking for the opened camera
        camera = cv2.VideoCapture(i)
        if camera.isOpened():
            print(f"Camera with index {i} is available and will be used")
            # camera.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # not working directly
            camera.set(cv2.CAP_PROP_GAIN, 0.75)
            camera_handle = camera; break
        else:
            print(f"Camera with index {i} is not available")

    # Read a few images and show them using pyplot
    if camera_handle is not None and camera_handle.isOpened():
        for i in range(4):
            camera.set(cv2.CAP_PROP_GAIN, 0.25*(i+1))
            read_flag, frame = camera_handle.read()  # read single frame
            if read_flag:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # required conversion from BGR to RGB
                plt.figure(f"Image with Gain: {0.25*(i+1)}")
                plt.imshow(frame); plt.axis("off"); plt.tight_layout()

        # Release camera and some standard camera
        camera_handle.release(); cv2.destroyAllWindows()