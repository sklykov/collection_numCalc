# coding=utf-8
"""
Fitting parameters of paraboloid over 2 dimensions.

@author: sklykov

@license: The Unlicense

"""
# %% Global imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')  # universally works for PyCharm and Spyder for setting interactive plotting


# %% Func. def-s
def paraboloid(coefficients: np.ndarray, coordinates: tuple) -> float:
    x, y = coordinates[0], coordinates[1]
    a, b, c, d, e, f = coefficients[0], coefficients[1], coefficients[2],  coefficients[3], coefficients[4], coefficients[5]
    return a*x*x + b*y*y + c*x*y + d*x + e*y + f

# Run
if __name__ == "__main__":
    plt.close("all")
    coeffs = [-1.4, -0.9, 1.3, 1.6, -0.8, 1.6]  # paraboloid parameters
    x = np.round(np.arange(start=-0.5, stop=0.6, step=0.1), 2)
    y = np.round(np.arange(start=-0.5, stop=0.6, step=0.1), 2)
    x_coords, y_coords = np.meshgrid(x, y)
    z = np.empty_like(x_coords.ravel())
    i = 0
    for x_c, y_c in zip(x_coords.ravel(), y_coords.ravel()):
        z[i] = paraboloid(coeffs, (x_c, y_c)); i += 1
    z = np.round(z.reshape(x_coords.shape), 4)
    # Code above produces same result as:
    # z = np.empty_like(x_coords)
    # for i in range(x_coords.shape[0]):
    #     for j in range(x_coords.shape[1]):
    #         z[i, j] = paraboloid(coeffs, (x_coords[i, j], y_coords[i, j]))
    # z = np.round(z, 4)

    fig = plt.figure(); axes = fig.add_subplot(projection='3d')
    axes.plot_surface(x_coords, y_coords, z, cmap='viridis'); plt.tight_layout()
    plt.show()
