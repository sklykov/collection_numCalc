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

# %% Simulation
diffusion_sim = diffusion([i_initial, j_initial])
even_bead = image_beads(character_size=bead_diameter)
scene = u_scene(width, height)
i_shift = i_initial
j_shift = j_initial

for i in range(2):
    even_bead.get_whole_shifted_blurred_bead(i_shift, j_shift, bead_intensity, NA, wavelength, calibration)
    # even_bead.plot_bead()
    i_shift = even_bead.offsets[0]
    j_shift = even_bead.offsets[1]
    print(i_shift, j_shift)
    scene.add_mask(i_shift - int(even_bead.height/2), j_shift - int(even_bead.width/2), even_bead.bead_img)
    scene.plot_image()
    scene.clear_scene()
    diffusion_sim.get_next_point()
    [i_shift, j_shift] = diffusion_sim.get_next_point()
