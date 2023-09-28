# -*- coding: utf-8 -*-
"""
Testing of parallel computation possibility using OpenCL capabilities.

@author: sklykov
@license: The Unlicense

"""
# %% Imports
import LoaderFile  # load sample image stored in the repository
import numpy as np
from numpy.fft import fftshift
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from time import perf_counter
plt.close('all')

opencl_installed = False
try:
    import pyopencl as cl  # The library for accessing OpenCL capabilities on any GPU supporting it
    import pyopencl.array as cla
    opencl_installed = True
except ModuleNotFoundError:
    print("Install pyopencl from conda-forge channel")
pyvkfft_installed = False
try:
    from pyvkfft.fft import fftn as vkfft  # The library for accessing fft function dispatched through OpenCL (or other interface) to the GPU
    pyvkfft_installed = True
except ModuleNotFoundError:
    print("Install pyvkfft from conda-forge channel")

# %% Tests
sample_img = LoaderFile.loadSampleImage()
sample_img_for_fft = sample_img.astype(np.complex64)  # required for errorless FFT transform
noisy_samples = []; h_sample, w_sample = sample_img.shape; max_val = np.max(sample_img)
n_test_samples = 10
# For guarantee uniqueness of sample images by adding noise to the initial sample image
for i in range(n_test_samples):
    noisy_sample_img = np.copy(sample_img); noisy_sample_img = noisy_sample_img.astype(dtype=np.uint16)
    noisy_sample_img += np.random.randint(max_val//15, high=max_val//2.5, size=(h_sample, w_sample), dtype='uint16')
    noisy_sample_img_for_fft = noisy_sample_img.astype(np.complex64)
    noisy_samples.append(noisy_sample_img_for_fft)

# Printing some info about available devices, check that OpenCL device is available
device_available = False
if opencl_installed:
    cl_platforms = cl.get_platforms()
    if len(cl_platforms) > 0:
        print("OpenCL Platforms:", cl_platforms)
        for platform in cl_platforms:
            cl_devices = platform.get_devices()
            if len(cl_devices) > 0:
                print("Devices on Platform:", cl_devices)
                device_available = True

# Select device for work, initializing required parameters
if device_available:
    cl_device = cl_devices[0]  # by default, select 1st available device

# Following the examples from https://pyvkfft.readthedocs.io/en/latest/examples/pyvkfft-fft.html
if device_available:
    cl_context = cl.Context(devices=(cl_device, ))
    cl_cq = cl.CommandQueue(cl_context)
    t1 = perf_counter()
    cl_array = cla.to_device(cl_cq, sample_img_for_fft)
    cl_array = vkfft(cl_array)
    t2 = perf_counter()
    print("Initial FFT transform using OpenCL ms:", int(round(1000.0*(t2-t1), 0)))
    plt.figure(); plt.imshow(noisy_sample_img); plt.tight_layout()  # Check noisy sample generation
    plt.figure(); plt.imshow(fftshift(np.abs(cl_array.get())), norm=LogNorm()); plt.tight_layout()  # Check FFT result
    # Simple measure of FFT performance
    t_mean_ms = 0
    for i in range(n_test_samples):
        t1 = perf_counter()
        cl_array = cla.to_device(cl_cq, noisy_samples[i])
        cl_array = vkfft(cl_array)
        t2 = perf_counter()
        t_mean_ms += 1000.0*(t2-t1)
    t_mean_ms = int(np.round(t_mean_ms / n_test_samples, 0))
    print(f"FFT using OpenCL of ({w_sample}x{h_sample} image) ms measured on {n_test_samples} samples:", t_mean_ms)
    plt.figure(); plt.imshow(fftshift(np.abs(cl_array.get())), norm=LogNorm()); plt.tight_layout()  # Check FFT result of last noisy sample
