# -*- coding: utf-8 -*-
"""
Test calculated values (compare, checking precision, etc.).

@author: ssklykov
"""
# %% Imports
import os
import numpy as np

# %% Evalution of importance on number of integration steps for calculation of integrals on sub-apertures of Zernike pols (-3, 3), (-1, 3)
calibration_folder = os.path.dirname(__file__)
Z20_file_name = "ZSHwfs20[(-3, 3), (-1, 3)].npy"; Z20_file_path = os.path.join(calibration_folder, Z20_file_name)
Z20 = np.load(Z20_file_path)
Z10_file_name = "ZSHwfs10[(-3, 3), (-1, 3)].npy"; Z10_file_path = os.path.join(calibration_folder, Z10_file_name)
Z10 = np.load(Z10_file_path)
# for accessing the influence on the precision calculations of decreasing number of integration steps
diff_Z20_Z10 = np.round(np.absolute(Z20 - Z10), 5)
Z40_file_name = "ZSHwfs40[(-3, 3), (-1, 3)].npy"; Z40_file_path = os.path.join(calibration_folder, Z40_file_name)
Z40 = np.load(Z40_file_path)
diff_Z40_Z20 = np.round(np.absolute(Z40 - Z20), 5)
Z60_file_name = "ZSHwfs60[(-3, 3), (-1, 3)].npy"; Z60_file_path = os.path.join(calibration_folder, Z60_file_name)
Z60 = np.load(Z60_file_path); Z3313 = np.load(Z60_file_path)
diff_Z60_Z40 = np.round(np.absolute(Z60 - Z40), 5)
Z80_file_name = "ZSHwfs80[(-3, 3), (-1, 3)].npy"; Z80_file_path = os.path.join(calibration_folder, Z80_file_name)
Z80 = np.load(Z80_file_path)
diff_Z80_Z60 = np.round(np.absolute(Z80 - Z60), 5)
print("Max difference in integral values between sequential # of steps used for their evaluation:")
print("10 -> 20 steps: ", np.max(diff_Z20_Z10))
print("20 -> 40 steps: ", np.max(diff_Z40_Z20))
print("40 -> 60 steps: ", np.max(diff_Z60_Z40))
print("60 -> 80 steps: ", np.max(diff_Z80_Z60))

# %% Similar evalution for another Zernike polynomials (0, 4), (2, 4) => similar behaviour of reducing difference between integrals
# Z20_file_name = "ZSHwfs20[(0, 4), (2, 4)].npy"; Z20_file_path = os.path.join(calibration_folder, Z20_file_name)
# Z20 = np.load(Z20_file_path)
# Z10_file_name = "ZSHwfs10[(0, 4), (2, 4)].npy"; Z10_file_path = os.path.join(calibration_folder, Z10_file_name)
# Z10 = np.load(Z10_file_path)
# print("******************************************************************************************")
# print("Max difference in integral values between sequential # of steps used for their evaluation:")
# print("10 -> 20 steps: ", np.max(diff_Z20_Z10))

# %% Compose already calculated values
Z1_name = "ZSHwfs[(-1, 1), (1, 1), (-2, 2), (0, 2), (2, 2)].npy"; Z1_path = os.path.join(calibration_folder, Z1_name)
Z2_name = "ZSHwfs[(-3, 3), (-1, 3), (1, 3), (3, 3)].npy"; Z2_path = os.path.join(calibration_folder, Z2_name)
Z3_name = "ZSHwfs[(-4, 4), (-2, 4), (0, 4), (2, 4), (4, 4)].npy"; Z3_path = os.path.join(calibration_folder, Z3_name)
Z1 = np.load(Z1_path); Z2 = np.load(Z2_path); Z3 = np.load(Z3_path)
Z_overall_name = "Calibration14ZernikesShH.npy"; Z_overall_path = os.path.join(calibration_folder, Z_overall_name)
if not(os.path.exists(Z_overall_path)):
    Z_overall = np.concatenate((Z1, Z2, Z3), axis=1)
    np.save(Z_overall_path, Z_overall)
