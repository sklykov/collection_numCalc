# -*- coding: utf-8 -*-
"""
Generate a few relevant samples using 'fluoscenepy' and 'zernpy' modules.

@author: sklykov
"""
# %% Global Imports
import numpy as np
import matplotlib.pyplot as plt
from fluoscenepy import force_precompilation, UscopeScene


# %% Notes
# TODO: 1) add to "get_random_objects" more printout hints if many objects are requisted without acceleration;
# 2) UscopeScene.get_random_objects: add pre-check intensity_range and image_type compatibility (needed for many objects generation)
# 3) some issue in placing algorithm - takes too long for the obvious task of placing small objects on the large picture


# %% Parameters

# %% Script run
if __name__ == "__main__":
    plt.close("all")
    force_precompilation()  # to speed up all functions in fluoscenepy
    # uscene = UscopeScene(width=2048, height=2048, image_type=np.uint16)  # pixel sizes for Orca and some pco.edge cameras
    uscene = UscopeScene(width=1200, height=1080, image_type=np.uint16)  # some downscaled non-squared image for testing all algorithms
    fl_objs = UscopeScene.get_random_objects(mean_size=(28, 21), size_std=(12, 8), shapes='mixed', intensity_range=(2500, 4000),
                                             image_type=uscene.img_type, n_objects=2, verbose_info=True, accelerated=True)
    uscene.set_random_places(fl_objs, overlapping=False, touching=False, only_within_scene=False, verbose_info=True)
    uscene.put_objects_on(fl_objs, save_only_objects_inside=True); uscene.show_scene()
