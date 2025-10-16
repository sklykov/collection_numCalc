# coding=utf-8
"""
Evaluate performance of OpenCV filter applying methods and comparing with standard scikit-image ones.

@author: sklykov
@license: The Unlicense

"""
import cv2
import numpy as np
from LoaderFile import loadSampleImage
from skimage.filters import gaussian, laplace, difference_of_gaussians
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sample_img = loadSampleImage()
    plt.close("all")
    truncate = 3.0   # used for calculation of kernel sizes 2*np.ceil(truncate*sigma) + 1
    # Evaluate standard scikit-image consecutive methods calls
    print("**** scikit-image LoG evaluation ****")
    t1 = time.perf_counter(); n_calculations = 8
    for i in range(n_calculations):
        sigma = 1.0 + 0.25*i
        img_g = gaussian(sample_img.copy().astype(np.float64), sigma=sigma, truncate=truncate)
        # Meaning of np.ceil(1.25) -> 2 - rounds to higher integer
        img_l = laplace(img_g)  # actually, 3x3 is standard template that not scaled further by increasing ksize
        # Plot 1 image for comparison with OpenCV output
        if i == n_calculations-2:
            plt.figure(f"skimage Gaussian with sigma: {sigma}"); plt.imshow(img_g, cmap=plt.cm.gray); plt.tight_layout(); plt.axis('off')
            plt.figure(f"skimage Laplacian with sigma: {sigma}"); plt.imshow(np.abs(img_l), cmap=plt.cm.gray)
            plt.tight_layout(); plt.axis('off')
    print(f"The scikit-image methods requires for #{n_calculations} LoG calculations ms:", int(np.round(1E3*(time.perf_counter() - t1))))
    print("For single LoG computation it takes ~ ms:", int(np.round((1E3*(time.perf_counter() - t1))/n_calculations)))

    # Evaluate OpenCV methods
    # First, check and set number of threads for OpenCV methods
    print("\n**** OpenCV LoG evaluation ****")
    print("Detected by OpenCV Threads:", cv2.getNumThreads())
    cv2.setNumThreads(cv2.getNumThreads() - 1)  # leave 1 thread for other tasks (potentially, GUI handling)
    print("OpenCV uses optimized methods:", cv2.useOptimized())
    if not cv2.useOptimized():
        cv2.setUseOptimized(True)
    t1 = time.perf_counter()
    for i in range(n_calculations):
        sigma = 1.0 + 0.25*i
        kernel_size = int(2*np.ceil(truncate*sigma) + 1)
        img_g = cv2.GaussianBlur(src=sample_img.copy().astype(np.float64), ksize=(kernel_size, kernel_size), sigmaX=sigma)
        img_l = cv2.Laplacian(src=img_g, ksize=3, ddepth=cv2.CV_64F)  # make it equal to the used by skimage
        # Plot 1 image for comparison with scikit-image output
        if i == n_calculations - 2:
            plt.figure(f"OpenCV Gaussian with sigma: {sigma}"); plt.imshow(img_g, cmap=plt.cm.gray); plt.tight_layout(); plt.axis('off')
            plt.figure(f"OpenCV Laplacian with sigma: {sigma}"); plt.imshow(np.abs(img_l), cmap=plt.cm.gray)
            plt.tight_layout(); plt.axis('off')
    print(f"The OpenCV methods requires for #{n_calculations} LoG calculations ms:", int(np.round(1E3 * (time.perf_counter() - t1))))
    print("For single LoG computation it takes ~ ms:", int(np.round((1E3 * (time.perf_counter() - t1)) / n_calculations)))

    # Evaluate DoG filter speed implemented by single threaded skimage method
    print("\n**** Evaluate DoG performance ****")
    t1 = time.perf_counter()
    for i in range(n_calculations):
        sigma = 1.0 + 0.25 * i; s2 = round(1.6*sigma, 4)
        img_l = difference_of_gaussians(sample_img.copy().astype(np.float64), low_sigma=sigma, high_sigma=s2)
        # Plot 1 image for comparison with OpenCV output
        if i == n_calculations - 2:
            plt.figure(f"skimage DoG with sigma: {sigma}"); plt.imshow(np.abs(img_l), cmap=plt.cm.gray)
            plt.tight_layout(); plt.axis('off')
    print(f"The scikit-image DoG requires for #{n_calculations} LoG calculations ms:", int(np.round(1E3 * (time.perf_counter() - t1))))
    print("For single DoG computation it takes ~ ms:", int(np.round((1E3 * (time.perf_counter() - t1)) / n_calculations)))

    plt.show()  # needed for PyCharm run for retain interactive plots
    # Dev. Note: now, the parameters for skimage and OpenCV selected in the way, that results are similar (visually)
