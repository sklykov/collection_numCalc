# -*- coding: utf-8 -*-
"""
Evaluation of DoG filtering by accounting non-zero pixels after applying it.

@author: sklykov
@license: The Unlicense

"""
# %% Imports
from LoaderFile import loadSampleImage  # loading sample from 'resources' folder
import matplotlib.pyplot as plt
from skimage.filters import difference_of_gaussians
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.morphology import disk
from skimage.filters.rank import mean, median
import numpy as np

# %% Parameters
low_sigma1 = 0.25  # defines smoothing of small noisy details / edges (noise or finest details)
# high_sigma1 = 4.0  # defines the effective length of depicted edge on the resulting image
high_sigma1 = round(1.6*low_sigma1, 4)  # default value for making the effect of DoG as the effect of LoG
# Hint for the sigma selection for DoG:
# https://en.wikipedia.org/wiki/Difference_of_Gaussians#Details_and_applications
# https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.difference_of_gaussians

visualize_dog_images = False  # 1st attempt to evaluate the concept
define_largest_sigma = True  # loop for defining the highest output evaluation

# %% Run as the script
if __name__ == "__main__":
    if not plt.isinteractive():
        plt.ion()
    # Plotting the original sample
    plt.close("all"); sample_img = loadSampleImage(); h, w = sample_img.shape; ratio = h/w; w_fig = 7.2
    plt.figure("Sample", figsize=(w_fig, w_fig*ratio)); plt.imshow(sample_img, cmap=plt.cm.gray); plt.tight_layout()

    if visualize_dog_images:
        # Show the result of applying filters
        img_dog1 = difference_of_gaussians(sample_img, low_sigma=low_sigma1, high_sigma=high_sigma1)
        img_dog1 = np.where(img_dog1 > 0.0, img_dog1, 0.0)  # saves only the positive values on an image
        img_dog1 = rescale_intensity(img_as_ubyte(img_dog1))  # convert to U8 and rescaling the intensity
        plt.figure(f"DoG: {low_sigma1}, {high_sigma1}", figsize=(w_fig, w_fig*ratio))
        plt.imshow(img_dog1, cmap=plt.cm.gray); plt.tight_layout()
        print("DoG: Counted non-zero pixel values/number of pixels:", round(np.sum(img_dog1)/(h*w), 6))

        # Testing the same workflow on the pre-smoothed image
        mask_r = 4; smoothing_mask = disk(mask_r)
        smoothed_sample_mean = mean(sample_img, smoothing_mask); smoothed_sample_median = median(sample_img, smoothing_mask)
        plt.figure(f"Smoothed Sample: Mean Disk({mask_r})", figsize=(w_fig, w_fig*ratio))
        plt.imshow(smoothed_sample_mean, cmap=plt.cm.gray); plt.tight_layout()
        plt.figure(f"Smoothed Sample: Median Disk({mask_r})", figsize=(w_fig, w_fig*ratio))
        plt.imshow(smoothed_sample_median, cmap=plt.cm.gray); plt.tight_layout()

        img_dog_mean = difference_of_gaussians(smoothed_sample_mean, low_sigma=low_sigma1, high_sigma=high_sigma1)
        img_dog_mean = np.where(img_dog_mean > 0.0, img_dog_mean, 0.0); img_dog_mean = rescale_intensity(img_as_ubyte(img_dog_mean))
        plt.figure(f"DoG Mean: {low_sigma1}, {high_sigma1}", figsize=(w_fig, w_fig*ratio))
        plt.imshow(img_dog_mean, cmap=plt.cm.gray); plt.tight_layout()
        print("Mean DoG: Counted non-zero pixel values/number of pixels:", round(np.sum(img_dog_mean)/(h*w), 6))

        img_dog_median = difference_of_gaussians(smoothed_sample_median, low_sigma=low_sigma1, high_sigma=high_sigma1)
        img_dog_median = np.where(img_dog_median > 0.0, img_dog_median, 0.0); img_dog_median = rescale_intensity(img_as_ubyte(img_dog_median))
        plt.figure(f"DoG Median: {low_sigma1}, {high_sigma1}", figsize=(w_fig, w_fig*ratio))
        plt.imshow(img_dog_median, cmap=plt.cm.gray); plt.tight_layout()
        print("Median DoG: Counted non-zero pixel values/number of pixels:", round(np.sum(img_dog_median)/(h*w), 6))

    if define_largest_sigma:
        estimations = []  # for storing calculated effect estimations of DoG filtering
        # Presmoothing initial image for suppression of the sharp pixel noise
        mask_r = 1; smoothing_mask = disk(mask_r)
        sample_img = median(sample_img, smoothing_mask)  # median noise filtering
        plt.figure("Smoothed Sample", figsize=(w_fig, w_fig*ratio)); plt.imshow(sample_img, cmap=plt.cm.gray); plt.tight_layout()
        sigmas = np.round(np.arange(start=0.5, stop=6.6, step=0.1), 3)
        for sigma in sigmas:
            img_dog = difference_of_gaussians(sample_img, low_sigma=sigma)
            img_dog_pos = np.where(img_dog > 0.0, 1, 0)
            estimations.append(round(np.sum(img_dog_pos)/(h*w), 6))  # sum of non-zero pixels (more borders or details => more non-zero pixels)
        # Plot max effective sigma (based on sum of non-zero pixels - detected edges of an image)
        max_est = np.max(estimations)
        if max_est < 1.0:
            estimations = list(2.0*np.asarray(estimations))
        plt.figure("Effectiveness Estimation (non-zero pixels) of DoG filtering vs Sigma")
        plt.plot(sigmas, estimations, 'o'); plt.tight_layout(); i_max = estimations.index(np.max(estimations))
        img_dog = difference_of_gaussians(sample_img, low_sigma=sigmas[i_max])
        img_dog = np.where(img_dog > 0.0, img_dog, 0.0); img_dog = rescale_intensity(img_as_ubyte(img_dog))
        plt.figure(f"DoG Image with Max Estimation, Sigma: {sigmas[i_max]}", figsize=(w_fig, w_fig*ratio))
        plt.imshow(img_dog, cmap=plt.cm.gray); plt.tight_layout()
