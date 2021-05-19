# -*- coding: utf-8 -*-
"""
Simulation of dynamics of the fluorescent bead undergoing diffusion with specified parameters

@author: ssklykov
"""
# %% Import section
from u_scene import u_scene
from fluorescent_beads import image_beads
from dynamics import diffusion

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
nFrames = 1000

# %% Simulation
diffusion_sim = diffusion([i_initial, j_initial], D=diffusion_coeff)
even_bead = image_beads(character_size=bead_diameter)
# print(even_bead.height, even_bead.width)
scene = u_scene(width, height)
i_center = i_initial
j_center = j_initial
# For fixing the possible discrepancies between origin of picture (the origin of bead picture) and its center
# This discrepancy could emerge during generation of next position f bead's center and origin coordinates
# i_origin = i_center - ((even_bead.height-1)//2)
# j_origin = j_center - ((even_bead.width-1)//2)
# print(even_bead.height, even_bead.width)
# print(i_origin, j_origin)
i_step = 0.0
j_step = 0.0
# print("i step:", i_step, "j step:", j_step)


for i in range(nFrames):
    even_bead.get_whole_shifted_blurred_bead(i_center, j_center, bead_intensity, NA, wavelength, calibration)
    # even_bead.plot_bead()
    i_center = even_bead.offsets[0]  # recalculate even pixel offset for adding the bead to the scene (background)
    j_center = even_bead.offsets[1]  # recalculate even pixel offset for adding the bead to the scene (background)

    # HINT: below calculations are kept for reference if it may be needed to check again the calculations
    # i_origin_old = i_center - int(even_bead.height/2)
    # j_origin_old = j_center - int(even_bead.width/2)
    # print("i center:", i_center, "j center:", j_center)
    # print("i old:", i_origin_old, "j old:", j_origin_old)
    # print(even_bead.height_changed, even_bead.width_changed)
    # New way for calculation of i_origin and j_origin - to prevent the issue of emerging discrepency between
    # coordinates of bead's center (i_center, j_center) and origin of its image (i_origin, j_origin) for further drawing
    i_origin = diffusion.get_i_origin(i_center, even_bead.height)
    j_origin = diffusion.get_i_origin(j_center, even_bead.width)
    # print("i new:", i_origin, "j new:", j_origin)
    # print()

    # Placing calculated image of a bead into background (scene)
    scene.add_mask(i_origin, j_origin, even_bead.bead_img)
    # scene.plot_image()
    scene.save_scene("png")
    scene.clear_scene()  # clearing generated bead for refreshing scene for the next frame
    if i < (nFrames-1):
        [i_center, j_center] = diffusion_sim.get_next_point()
        # i_step = diffusion_sim.x_generated_steps[len(diffusion_sim.x_generated_steps)-1]
        # j_step = diffusion_sim.y_generated_steps[len(diffusion_sim.y_generated_steps)-1]
        # print("i step:", i_step, "j step:", j_step)
    # print(i_center, j_center)
diffusion_sim.save_generated_stat(save_figures=True)
even_bead.save_used_parameters()
