# -*- coding: utf-8 -*-
"""
Test calculated values (compare, checking precision, etc.).

@author: ssklykov
"""
# %% Imports
import os
import numpy as np

# %% Evalution of importance on number of integration steps for calculation of integrals of Zernike pols (-3, 3), (-1, 3)
calibration_folder = os.path.dirname(__file__)
Z_steps_name = "ZSHwfs_various_steps[(-3, 3), (-1, 3)].npy"; Z_steps_path = os.path.join(calibration_folder, Z_steps_name)
if (os.path.exists(Z_steps_path)):
    Zsteps = np.load(Z_steps_path)
    Z10 = Zsteps[:, 0:4]; Z20 = Zsteps[:, 4:8]; Z40 = Zsteps[:, 8:12]; Z60 = Zsteps[:, 12:16]; Z80 = Zsteps[:, 16:20]
    diff_Z20_Z10 = np.round(np.absolute(Z20 - Z10), 5); diff_Z40_Z20 = np.round(np.absolute(Z40 - Z20), 5)
    diff_Z60_Z40 = np.round(np.absolute(Z60 - Z40), 5); diff_Z80_Z60 = np.round(np.absolute(Z80 - Z60), 5)
    print("Max difference in integral values between sequential # of steps used for their evaluation:")
    print("10 -> 20 steps: ", np.max(diff_Z20_Z10)); print("20 -> 40 steps: ", np.max(diff_Z40_Z20))
    print("40 -> 60 steps: ", np.max(diff_Z60_Z40)); print("60 -> 80 steps: ", np.max(diff_Z80_Z60))

# %% Similar evalution for another Zernike polynomials (0, 4), (2, 4) => similar behaviour of reducing difference between integrals

# %% Compose already calculated values
Z1_name = "ZSHwfs60[(-1, 1), (1, 1)].npy"; Z1_path = os.path.join(calibration_folder, Z1_name)
Z2_name = "ZSHwfs60[(-2, 2), (0, 2), (2, 2)].npy"; Z2_path = os.path.join(calibration_folder, Z2_name)
Z3_name = "ZSHwfs60[(-3, 3), (-1, 3), (1, 3), (3, 3)].npy"; Z3_path = os.path.join(calibration_folder, Z3_name)
Z4_name = "ZSHwfs50[(-4, 4), (-2, 4), (0, 4), (2, 4), (4, 4)].npy"; Z4_path = os.path.join(calibration_folder, Z4_name)
Z1 = np.load(Z1_path); Z2 = np.load(Z2_path); Z3 = np.load(Z3_path); Z4 = np.load(Z4_path)
# diffZcol = Z3[:, 2] - Z3[:, 5]
Z_overall_name = "Calibration14ZernikesShHMoreSteps.npy"; Z_overall_path = os.path.join(calibration_folder, Z_overall_name)
if not(os.path.exists(Z_overall_path)):
    Z_overall = np.concatenate((Z1, Z2, Z3, Z4), axis=1)
    np.save(Z_overall_path, Z_overall)
