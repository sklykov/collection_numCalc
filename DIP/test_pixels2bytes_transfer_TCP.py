# -*- coding: utf-8 -*-
"""
Testing the size in bytes of converted raw pixel data from a sample image.

@author: sklykov
@license: The Unlicense
"""
# %% Global imports
import numpy as np
import skimage
from pathlib import Path
import sys
# import matplotlib.pyplot as plt
import socket
import time
from threading import Thread

# %% Global parameters
default_port = 8098; timeout_connection_s = 2.5; max_iterations = 2; recived_image = None

# %% Test converting of an image and usage of bytes()
path_sample = Path(__file__).cwd().joinpath("resources").joinpath("nesvizh_grey.jpg")
img = skimage.io.imread(path_sample, as_gray=True)  # 'uint8' numpy array

# Conversion of uint8 image
b_img = bytes(img); h, w = img.shape; img_row = img[1, :]; b_img_row = bytes(img_row)
print(f"Whole image {w}x{h} {img.dtype} size in bytes:", sys.getsizeof(b_img))
print(f"Image row with width={w} and type '{img.dtype}' size in bytes:", sys.getsizeof(b_img_row))

# Conversion of uint16 image
scaling_coeff = int(round((2**16 - 1)/np.max(img), 0))
img_uint16 = np.copy(img).astype(dtype=np.uint16)*scaling_coeff
# plt.figure("16 bit converted sample", figsize=(7.4, 4.8)); plt.imshow(img_uint16, cmap=plt.cm.gray, origin='upper')
# plt.axis('off'); plt.tight_layout()
b_img16 = bytes(img_uint16); img_row16 = img_uint16[1, :]; b_img_row16 = bytes(img_row16)
print(f"Whole image {w}x{h} {img_uint16.dtype} size in bytes:", sys.getsizeof(b_img16))
print(f"Image row with width={w} and type '{img_uint16.dtype}' size in bytes:", sys.getsizeof(b_img_row16))


# %% Test the speed of transfer an image through TCP connection
def launch_server(port: int = 8096, timeout_connection_sec: int = 3.0):
    """
    Launch a server on localhost and specified network port.

    Parameters
    ----------
    port : int, optional
        Network port for establishing TCP connection. The default is 8096.
    timeout_connection_sec : int, optional
        Maximum timeout in seconds for establishing the connection with a client. The default is 3.0.
    Returns
    -------
    None.

    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout_connection_sec)  # prevent to wait too long for some client connected
        s.bind(('localhost', port)); s.listen()
        try:
            (connection, address) = s.accept()
            event_loop = True  # for command processing loop
            with connection:
                while event_loop:
                    try:
                        data = connection.recv(1024); command = data.decode('utf-8')
                        print("Received from a client:", command)
                        # Ping command - for checking the connection
                        if "Ping" in command:
                            connection.send(b'Echo from the local server')
                        # Some 'Command 1' received (e.g., button clicked)
                        elif "Image" in command:
                            size_to_receive = int(command[5:]); print("Awaited image size:", size_to_receive)
                            t1 = time.perf_counter()
                            image = connection.recv(size_to_receive)
                            img = np.asarray(list(image)).reshape(h, w)
                            print("Properties of the received image:", img.shape, np.max(img), np.min(img), flush=True)
                            print("Receiving image takes ms:", round(1000*(time.perf_counter()-t1), 0), flush=True)
                            # print("Image:", list(image), flush=True)
                            connection.send(b'Image successfully received')
                        elif "Quit" in command:
                            time.sleep(2.0)  # imitation of the work
                            connection.send(b'Server closed'); event_loop = False
                        # One of way of checking preserved connection - check that the received command not empty
                        if len(command) == 0:
                            print("The empty string received from the client, most likely it finished working")
                            event_loop = False
                    except (ConnectionAbortedError, ConnectionRefusedError):
                        print("Client has closed the connection")
                        event_loop = False; break
        except (ConnectionRefusedError, TimeoutError):
            print("No connection with a client established on port:", port)


def launch_client(port: int = 8096, timeout_connection_sec: int = 2.0):
    """
    Launch a client on localhost and specified network port.

    Parameters
    ----------
    port : int, optional
        Network port for establishing TCP connection. The default is 8096.
    timeout_connection_sec : int, optional
        Maximum timeout in seconds for establishing the connection with a server. The default is 2.0.

    Returns
    -------
    None.

    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout_connection_sec)
        try:
            connection.connect(('localhost', port))  # Try to connect to the listening server
            # Send 'Ping' command and get the answer
            time.sleep(0.1); connection.send(b"Ping")
            response = connection.recv(1024).decode('utf-8')
            print("Received back:", response)
            actions_loop = True; iteration = 0; quit_sent = False
            # Loop with actions
            while actions_loop:
                # Some command send to the server and awaiting the response from it
                if iteration == 0:
                    command = f"Image {sys.getsizeof(b_img)}"
                    connection.send(command.encode()); time.sleep(4/1000)
                    print("Image sizes:", img.shape)
                    t3 = time.perf_counter()
                    connection.sendall(b_img)
                    print("Sending image takes ms:", round(1000*(time.perf_counter()-t3), 0), flush=True)
                    time.sleep(1.0); iteration += 1
                else:
                    connection.send(b"Quit"); iteration += 1
                    quit_sent = True; actions_loop = False
                # Wait response from the server
                try:
                    data = connection.recv(1024)
                    # Check that data has been received
                    if len(data) > 0:
                        print("Received from a server:", str(data, 'utf-8'), flush=True)
                    else:
                        print("Empty response received from a server", flush=True)
                except ConnectionAbortedError:
                    print("Server have been closed / stoped working", flush=True)
                    actions_loop = False; break
                except TimeoutError:
                    print("Nothing received from a server, timeout", flush=True)
                # Making some delay between commands
                if iteration == 1:
                    time.sleep(timeout_connection_sec - 0.5)  # less than timeout
                # Check for exit from the loop
                if iteration > max_iterations:
                    actions_loop = False; break
            # Send 'Quit' command to the server, if not sent by the action above
            if not quit_sent:
                try:
                    connection.send(b'Quit'); time.sleep(0.005)
                    print(connection.recv().decode('utf-8'))
                except (ConnectionAbortedError, ConnectionRefusedError):
                    print("The connection has been closed by the server before sending 'Quit' command")
        except (ConnectionRefusedError, TimeoutError):
            print("No connection with a server established on port:", port); connection.close()


def launch():
    """
    Launch a server and a client connected with each other and communicating through TCP connection.

    Both server and client are opened in the dedicated thread.

    Returns
    -------
    None.

    """
    # Launch server on the separate thread
    server_thread = Thread(target=launch_server, args=(8091, timeout_connection_s+1.5)); server_thread.start()
    # Launch client on the separate thread
    client_thread = Thread(target=launch_client, args=(8091, timeout_connection_s-0.25)); client_thread.start()
    # Wait for all threads exit
    if client_thread.is_alive():
        client_thread.join()  # client should exit normally by itself
    if server_thread.is_alive():
        server_thread.join(timeout=2.0*timeout_connection_s)


# %% Launch as the main script
if __name__ == "__main__":
    launch()
