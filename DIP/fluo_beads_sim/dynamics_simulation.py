# -*- coding: utf-8 -*-
"""
Simulation of dynamics of the fluorescent bead undergoing diffusion with specified parameters.
Each new position is drawn on the new image that is further saved in the default folder.

@author: ssklykov
"""
# %% Import section
from u_scene import u_scene
from fluorescent_beads import image_beads
from dynamics import diffusion
# import math  # the used function: math.modf()
import numpy as np

# %% General parameters
width = 200
height = 200
# Initial point for the particle - center of the generated scene (picture or frame)
i_initial = int(height / 2)
j_initial = int(width / 2)
bead_intensity = 204  # % of maximal value of 255 for 8bit gray image
wavelength = 486
NA = 1.2
calibration = 111
bead_diameter = 28
diffusion_coeff = 1
nFrames = 10
debug = True
round_presicion = 6  # rounding of floats number calculations (e.g., coordinates)

# %% Simulation
diffusion_sim = diffusion([i_initial, j_initial], D=diffusion_coeff)
even_bead = image_beads(character_size=bead_diameter)
scene = u_scene(width, height)
i_center = i_initial
j_center = j_initial
i_step = 0.0
j_step = 0.0
i_next = i_initial
j_next = j_initial
# print("i step:", i_step, "j step:", j_step)
# print("initials", i_center, j_center)
origins = []
centers = []


for i in range(nFrames):
    even_bead.get_whole_shifted_blurred_bead(i_center, j_center, bead_intensity, NA, wavelength, calibration,
                                             round_presicion, debug=True)

    # Origin - for placing bead image
    [i_origin, j_origin] = even_bead.get_origin_coordinates()
    # diffusion_sim.save_bead_origin([i_origin, j_origin])  # for possible debugging of the program

    # HINT: commented out code below was intended for searching of reasons for error during placing a bead on a scene
    # (i_float_part, i_int_part) = math.modf(i_center)
    # i_float_part = round(i_float_part, round_presicion)
    # (j_float_part, j_int_part) = math.modf(j_center)
    # j_float_part = round(j_float_part, round_presicion)
    # if (i_float_part > 0.0) and (even_bead.height % 2 != 0):
    #     print("i center:", i_center, "origin:",  i_origin, "height:", even_bead.height, "float:", i_float_part)
    #     print("width:",  even_bead.width, "float:", j_float_part)
    # if (j_float_part > 0.0) and (even_bead.width % 2 != 0):
    #     print("j center:", j_center, "origin:", j_origin, "width:",  even_bead.width, "float:", j_float_part)
    #     print("height:", even_bead.height, "float:", i_float_part)

    # HINT: again, the code below (possibly commented out) for catching the bug with wrong placing of a bead
    i_center = even_bead.offsets[0]  # recalculate even pixel offset for adding the bead to the scene (background)
    j_center = even_bead.offsets[1]  # recalculate even pixel offset for adding the bead to the scene (background)
    # origins.append([i_origin, j_origin])
    # centers.append([i_center, j_center])
    # if i > 0:
    #     i_origin_step = origins[len(origins)-1][0] - origins[len(origins)-2][0]
    #     i_center_step = centers[len(centers)-1][0] - centers[len(centers)-2][0]
    #     j_origin_step = origins[len(origins)-1][1] - origins[len(origins)-2][1]
    #     j_center_step = centers[len(centers)-1][1] - centers[len(centers)-2][1]
    #     if i_origin_step != i_center_step:
    #         print("i origin step:", i_origin_step, "i center:", i_center_step)
    #     if j_origin_step != j_center_step:
    #         print("j origin step:", j_origin_step, "j center:", j_center_step)

    # Placing calculated image of a bead into background (scene)
    scene.add_mask(i_origin, j_origin, even_bead.bead_img, debug)

    # HINT: hunting for bugs during picture generation
    if (i > 0) and debug:
        i_mass_center = np.round(scene.centers_of_mass[i][0], round_presicion)
        if abs(i_next - i_mass_center) > 0.5:
            print("bug", "i_next", i_next, "i_mass", i_mass_center)
            print("i_origin", i_origin, "height", even_bead.height, "i_center", i_center)
            print("input i_offset", even_bead.debug_offsets[i][0], "calculated i_center", even_bead.debug_centers[i][0])
            even_bead.plot_bead()
        j_mass_center = np.round(scene.centers_of_mass[i][1], round_presicion)
        if abs(j_next - j_mass_center) > 0.5:
            print("bug", "j_next", j_next, "j_mass", j_mass_center)
            print("j_origin", j_origin, "width", even_bead.width, "j_center", j_center)
            print("input j_offset", even_bead.debug_offsets[i][1], "calculated j_center", even_bead.debug_centers[i][1])
            even_bead.plot_bead()

    # scene.plot_image()
    scene.save_scene("png")  # regulates of extension of generated images
    scene.clear_scene()  # clearing generated bead for refreshing scene for the next frame
    if i < (nFrames-1):
        [i_center, j_center] = diffusion_sim.get_next_point(round_presicion)
        # print("i center:", i_center, "i origin", i_origin, "j center:", j_center, "j origin", j_origin)
        i_next = i_center
        j_next = j_center
        # i_step = diffusion_sim.x_generated_steps[len(diffusion_sim.x_generated_steps)-1]
        # j_step = diffusion_sim.y_generated_steps[len(diffusion_sim.y_generated_steps)-1]

diffusion_sim.save_generated_stat(save_figures=True)
even_bead.save_used_parameters()
