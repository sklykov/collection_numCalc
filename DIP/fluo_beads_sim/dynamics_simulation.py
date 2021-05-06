# -*- coding: utf-8 -*-
"""
Simulation of dynamics of the fluorescent bead

@author: ssklykov
"""
# %% Import section
from u_scene import u_scene
from fluorescent_beads import image_beads
from dynamics import diffusion

# %% General parameters
width = 100
height = 100
i_initial = int(height / 2)
j_initial = int(width / 2)
bead_intensity = 128
wavelength = 532
NA = 1.25
calibration = 110
bead_diameter = 20
diffusion_coeff = 0.25
nFrames = 5

# %% Simulation
diffusion_sim = diffusion([i_initial, j_initial], D=diffusion_coeff)
even_bead = image_beads(character_size=bead_diameter)
scene = u_scene(width, height)
i_shift = i_initial
j_shift = j_initial

for i in range(nFrames):
    even_bead.get_whole_shifted_blurred_bead(i_shift, j_shift, bead_intensity, NA, wavelength, calibration)
    # even_bead.plot_bead()
    i_shift = even_bead.offsets[0]  # recalculate even pixel offset for adding the bead to the scene (background)
    j_shift = even_bead.offsets[1]  # recalculate even pixel offset for adding the bead to the scene (background)
    # print(i_shift, j_shift)
    # Below the conversion (i_shift - ...) is for converting center of bead coordinates to origin of the bead image
    scene.add_mask(i_shift - int(even_bead.height/2), j_shift - int(even_bead.width/2), even_bead.bead_img)
    # scene.plot_image()
    scene.save_scene()
    scene.clear_scene()  # clearing generated bead
    [i_shift, j_shift] = diffusion_sim.get_next_point()
diffusion_sim.save_generated_stat(save_figures=True)
