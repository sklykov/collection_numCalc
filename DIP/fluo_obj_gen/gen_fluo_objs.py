# -*- coding: utf-8 -*-
"""
Generate a few relevant samples using 'fluoscenepy' and 'zernpy' modules.

@author: sklykov
"""
# %% Global Imports
import numpy as np
import matplotlib.pyplot as plt
from fluoscenepy import UscopeScene
from pathlib import Path
from datetime import datetime
import time
import skimage.io as io

# %% Notes
# TODO: 1) add to "get_random_objects" more printout hints if many objects are requisted without acceleration;
# 2) UscopeScene.get_random_objects: add pre-check intensity_range and image_type compatibility (needed for many objects generation)
# 3) some issue in placing algorithm - takes too long for the obvious task of placing small objects on the large picture


# %% Parameters

# %% Script run
if __name__ == "__main__":
    plt.close("all")  # close all opened figures
    get_mixed_shaped_objs = False  # flag for long calculation of precisely shaped objects
    # Generate objects using fluoscenepy library
    # uscene = UscopeScene(width=2048, height=2048, image_type=np.uint16)  # standard pixel sizes for Orca and some pco.edge cameras
    uscene = UscopeScene(width=272, height=231, image_type=np.uint16)  # some downscaled non-squared image for testing all algorithms
    uscene.precompile_methods(True)  # precompile methods for acceleration
    if get_mixed_shaped_objs:
        fl_objs = uscene.get_objects_acc(mean_size=(12, 10), size_std=(3.5, 2.2), shapes='mixed', intensity_range=(4000, 12000),
                                         image_type=uscene.img_type, n_objects=85, verbose_info=True)
    else:
        fl_objs = uscene.get_round_objects(mean_size=10, size_std=2, intensity_range=(4000, 12000), n_objects=80, image_type=uscene.img_type)
    uscene.set_random_places(fl_objs, overlapping=False, touching=False, only_within_scene=True, verbose_info=True)
    uscene.put_objects_on(fl_objs, save_only_objects_inside=True); uscene.show_scene()
    # Saving generated scene
    current_folder = Path(__name__).parent.absolute()
    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M")
    gen_img_path = current_folder.joinpath("SrcImg_" + timestamp + ".tiff")
    io.imsave(gen_img_path, uscene.image, check_contrast=False)
