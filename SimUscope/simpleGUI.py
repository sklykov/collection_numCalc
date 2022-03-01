# -*- coding: utf-8 -*-
"""
Simple GUI for representing the simulated image.

@author: ssklykov
"""
# %% Imports
from generate_noise_pic import generate_noise_picture
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # import canvas container from matplotlib for tkinter
import tkinter
import matplotlib.pyplot as plt
import time
from threading import Thread
from queue import Queue, Empty
import numpy as np

# %% Some global variables
flag_generation = False  # For start/stop generation of some noisy pictures
messages = Queue(maxsize=5)

# %% Classes
class Refresher(Thread):
    def __init__(self, imshow_img, canvas, message_queue, refresh_delay_ms: int = 100):
        Thread.__init__(self)
        self.refresh_image = imshow_img
        self.canvas = canvas
        self.refresh_delay_ms = refresh_delay_ms
        self.message_queue = message_queue

    def run(self):
        global flag_generation
        i = 0
        while (flag_generation):
            t1 = time.time()
            # print("Refresher working")
            refresh_noise_image(self.refresh_image, self.canvas)
            time.sleep(self.refresh_delay_ms/1000)
            i += 1
            t2 = time.time(); print("Passed time for frame generation and show:", np.round((t2-t1)*1000, 0), "ms")
            if i > 500:
                self.message_queue.put_nowait(Exception("Refresher exception"))
                print(self.message_queue.qsize())
                raise Exception("Refresher exception")
        print("Refresher stopped")


# %% Functions
def refresh_noise_image(imshow_img, cnvs):
    """
    Refresh of the noisy image inside the canvas widget by clicking the button.

    Parameters
    ----------
    imshow_img : matplolib.image.AxesImage
        For updating its content just in place, using the function set_data(image).
    cnvs : FigureCanvasTkAgg from matplotlib.backends.backend_tkagg
        For forced redraw of its content.

    Returns
    -------
    None.

    """
    img = (generate_noise_picture(200, 200))  # generation noisy image
    imshow_img.set_data(img)  # update the content of AxesImage helpes to re-draw associated image in canvas widget
    cnvs.draw()  # Redraw the image in the canvas widget


def toggle_continuous_generation(imshow_img, canvas, messages, refresh_delay_ms: int = 100):
    global flag_generation
    flag_generation = not flag_generation
    print("Flag of generation:", flag_generation)
    if (flag_generation):
        image_refresher = Refresher(imshow_img, canvas, messages, refresh_delay_ms); image_refresher.start()


def encounter_exception(root_widget, message_queue):
    # print("Encounter is looking for exception")
    if (message_queue.qsize() > 0) and not(message_queue.empty()):
        try:
            e = message_queue.get_nowait()
        except Empty:
            pass
        if isinstance(e, Exception):
            print("DESTROYING MAIN APPLICATION")
            root_widget.destroy()
        if isinstance(e, str):
            # print(e, "- received and accounted")
            pass
    # Calling in the end itself another time for making infinite loop
    root_widget.after(300, encounter_exception, root_widget, message_queue)


def put_message(root_widget, message_queue):
    # print("Putting message")
    if (message_queue.qsize() < 3):
        message_queue.put_nowait("Message")
    root_widget.after(900, put_message, root_widget, message_queue)


# %% Tests
if __name__ == '__main__':
    root = tkinter.Tk(); root.title("Main window")  # top level widget
    img = (generate_noise_picture(200, 200))  # generation noisy image
    figure = plt.figure("pyplot"); axes_image = plt.imshow(img, cmap='gray')  # the links to 2 pyplot classes
    plt.axis("off"); plt.tight_layout()  # removing axis from image representation
    # Packing plot in the canvas widget
    canvas = FigureCanvasTkAgg(figure, master=root); canvas.draw(); canvas.get_tk_widget().pack()
    plt.close("pyplot")  # Close the additional independent matplolib plot in a window
    generate_button = tkinter.Button(root, text="Generate noisy image",
                                     command=(lambda: refresh_noise_image(imshow_img=axes_image, cnvs=canvas)))
    generate_button.pack(side=tkinter.LEFT)
    continuous_gen_button = tkinter.Button(root, text="Continuous generation",
                                           command=(lambda: toggle_continuous_generation(axes_image, canvas, messages, 2)))
    continuous_gen_button.pack(side=tkinter.RIGHT)
    root.after(300, encounter_exception, root, messages)  # Important: arguments - function, after - arguments. Thanks Stackoverflow!
    root.after(900, put_message, root, messages)

    root.mainloop()
    # Stopping the Refresher
    if flag_generation:
        flag_generation = False
        time.sleep(0.5)
