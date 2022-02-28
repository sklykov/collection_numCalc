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


# %% Tests
if __name__ == '__main__':
    root = tkinter.Tk(); root.title("Main window")  # top level widget
    img = (generate_noise_picture(200, 200))  # generation noisy image
    figure = plt.figure("pyplot"); axes_image = plt.imshow(img)  # the links to 2 pyplot classes
    plt.axis("off"); plt.tight_layout()  # removing axis from image representation
    # Packing plot in the canvas widget
    canvas = FigureCanvasTkAgg(figure, master=root); canvas.draw(); canvas.get_tk_widget().pack()
    plt.close("pyplot")  # Close the additional independent matplolib plot in a window
    generate_button = tkinter.Button(root, text="Generate noisy image",
                                     command=(lambda: refresh_noise_image(imshow_img=axes_image, cnvs=canvas)))
    generate_button.pack()

    root.mainloop()
