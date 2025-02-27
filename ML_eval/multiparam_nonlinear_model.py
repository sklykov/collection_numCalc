# -*- coding: utf-8 -*-
"""
Build and test non-linear multiparameters input-output model.

@author: sklykov
@license: The Unlicense

"""
# %% Imports
import numpy as np
import random

scikit_available = False
try:
    import sklearn
    if sklearn is not None:
        scikit_available = True
except ModuleNotFoundError:
    pass


# %% Model definition - e.g. 4 input parameters interconnected and resulted in 4 output, scaled responses
def nonlinear_multiparam_model(x: np.array) -> np.array:
    if len(x.shape) == 1 and len(x) == 4:
        y = np.zeros(shape=x.shape)  # initialize returning coefficients
        a0 = 1.11; a1 = 1.07; a2 = 1.05; a3 = 1.02  # non-linear coefficients
        percentage_diff = [0.27, 0.19, 0.08, 0.11]  # reduction of amplitude if several non-zero amplitudes provided (diff from 1.0)
        max_value_replicated = [2.37, 2.48, 1.89, 2.16]
        a = [a0, a1, a2, a3]  # storing in a list scaling coefficients
        # Single amplitude provided (checking)
        n_zero_el = 0; i_nonzero_el = 0; i_nonzero_els = []; i_zero_els = []
        # Checking number of almost zero and non-zero elements
        for i, el in enumerate(x):
            if el <= 0.01:
                n_zero_el += 1; i_zero_els.append(i)
            else:
                i_nonzero_el = i; i_nonzero_els.append(i)
        if n_zero_el == 3:
            # Single not zero element, construct non-linear response
            for i, el in enumerate(x):
                if i == i_nonzero_el:
                    if el < 0.5:
                        y[i] = a[i]*el + 0.01*random.randint(2, 8)*el
                    else:
                        calc_value = (a[i]-1.0)*el*el + a[i]*el + 0.01*random.randint(1, 7)*el
                        if calc_value < max_value_replicated[i]:
                            y[i] = calc_value
                        else:
                            y[i] = max_value_replicated[i]
                else:
                    y[i] = random.choice([0.0, 0.01, 0.02, 0.03])
        elif n_zero_el == 2:
            for i, el in enumerate(x):
                if i in i_zero_els:
                    y[i] = random.choice([0.0, 0.01, 0.02, 0.03])
                else:
                    if el < 0.5:
                        y[i] = (1.0 - percentage_diff[i])*(a[i]*el + 0.01*random.randint(2, 8)*el)
                    else:
                        calc_value = (1.0 - percentage_diff[i])*((a[i]-1.0)*el*el + a[i]*el + 0.01*random.randint(1, 7)*el)
                        if calc_value < max_value_replicated[i]:
                            y[i] = calc_value
                        else:
                            y[i] = max_value_replicated[i]
        elif n_zero_el == 3:
            if i in i_zero_els:
                y[i] = random.choice([0.0, 0.01, 0.02, 0.03])
            else:
                if el < 0.5:
                    y[i] = (0.8 - percentage_diff[i])*(a[i]*el + 0.01*random.randint(2, 8)*el)
                else:
                    calc_value = (0.8 - percentage_diff[i])*((a[i]-1.0)*el*el + a[i]*el + 0.01*random.randint(1, 7)*el)
                    if calc_value < max_value_replicated[i]:
                        y[i] = calc_value
                    else:
                        y[i] = max_value_replicated[i]
        else:
            if i in i_zero_els:
                y[i] = random.choice([0.0, 0.01, 0.02, 0.03])
            else:
                if el < 0.5:
                    y[i] = (0.6 - percentage_diff[i])*(a[i]*el + 0.01*random.randint(2, 8)*el)
                else:
                    calc_value = (0.6 - percentage_diff[i])*((a[i]-1.0)*el*el + a[i]*el + 0.01*random.randint(1, 7)*el)
                    if calc_value < max_value_replicated[i]:
                        y[i] = calc_value
                    else:
                        y[i] = max_value_replicated[i]
        return y
    else:
        print("This model only resolves 4 inputs to 4 outputs")
        return x


# %% Testing Model for input - output (Training data)
if __name__ == "__main__":
    x_input = []; y_data = []  # input and output data for testing ML regression
    # standard scanning scheme for getting data points
    x11 = [3.0, 0.0, 0.0, 0.0]; x12 = [1.5, 0.0, 0.0, 0.0]; x13 = [1.0, 0.0, 0.0, 0.0]; x14 = [0.5, 0.0, 0.0, 0.0]; x15 = [0.25, 0.0, 0.0, 0.0]
    x21 = [0.0, 3.0, 0.0, 0.0]; x22 = [0.0, 1.5, 0.0, 0.0]; x23 = [0.0, 1.0, 0.0, 0.0]; x24 = [0.0, 0.5, 0.0, 0.0]; x25 = [0.0, 0.25, 0.0, 0.0]
    x31 = [0.0, 0.0, 3.0, 0.0]; x32 = [0.0, 0.0, 1.5, 0.0]; x33 = [0.0, 0.0, 1.0, 0.0]; x34 = [0.0, 0.0, 0.5, 0.0]; x35 = [0.0, 0.0, 0.25, 0.0]
    x41 = [0.0, 0.0, 0.0, 3.0]; x42 = [0.0, 0.0, 0.0, 1.5]; x43 = [0.0, 0.0, 0.0, 1.0]; x44 = [0.0, 0.0, 0.0, 0.5]; x45 = [0.0, 0.0, 0.0, 0.25]
    x16 = [0.1, 0.0, 0.0, 0.0]; x26 = [0.0, 0.1, 0.0, 0.0]; x36 = [0.0, 0.0, 0.1, 0.0]; x46 = [0.0, 0.0, 0.0, 0.1]
    x_input.append(x11); x_input.append(x12); x_input.append(x13); x_input.append(x14); x_input.append(x15); x_input.append(x16)
    x_input.append(x21); x_input.append(x22); x_input.append(x23); x_input.append(x24); x_input.append(x25); x_input.append(x26)
    x_input.append(x31); x_input.append(x32); x_input.append(x33); x_input.append(x34); x_input.append(x35); x_input.append(x36)
    x_input.append(x41); x_input.append(x42); x_input.append(x43); x_input.append(x44); x_input.append(x45); x_input.append(x46)

    # Making random combinations of input parameters
    x1_max = 2.0; x2_max = 2.0; x3_max = 1.5; x4_max = 1.75
    for i in range(len(x_input)*2):
        zero_ampls = random.choice([0, 1, 2])  # choice how many zero amplitudes will be used in combinations
        zeroing_coefficients = [1.0, 1.0, 1.0, 1.0]
        if zero_ampls == 1:
            zero_ampl = random.choice([0, 1, 2])  # select zero coefficient
            zeroing_coefficients[zero_ampl] = 0.0
        elif zero_ampls == 2:
            zero_ampl1 = random.choice([0, 1, 2, 3]); indices = [0, 1, 2, 3]; del indices[zero_ampl1]
            zero_ampl2 = random.choice(indices)  # select 2nd zero coefficient
            zeroing_coefficients[zero_ampl1] = 0.0; zeroing_coefficients[zero_ampl2] = 0.0
        # Compose random amplitudes in combination
        x1 = zeroing_coefficients[0]*random.random()*x1_max
        x2 = zeroing_coefficients[1]*random.random()*x2_max
        x3 = zeroing_coefficients[2]*random.random()*x3_max
        x4 = zeroing_coefficients[3]*random.random()*x4_max
        x_row = [x1, x2, x3, x4]; x_input.append(x_row)  # add to the data

    x_input = np.asarray(x_input)

    # Getting output - Y
    for x_row in x_input:
        y = nonlinear_multiparam_model(x_row)
        y_data.append(y)
    y_data = np.asarray(y_data)

    # ML testing
    if scikit_available:
        # Testing kNN algorithm
        from sklearn import neighbors
        n_neighbors = 3
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
        knn_model = knn.fit(x_input, y_data)

        # Making test data
        x_test = []; x_test.append([0.0, 0.3, 0.0, 0.0])
        for i in range(10):
            zero_ampls = random.choice([0, 1, 2, 3])  # choice how many zero amplitudes will be used in combinations
            zeroing_coefficients = [1.0, 1.0, 1.0, 1.0]
            if zero_ampls == 1:
                zero_ampl = random.choice([0, 1, 2])  # select zero coefficient
                zeroing_coefficients[zero_ampl] = 0.0
            elif zero_ampls == 2:
                zero_ampl1 = random.choice([0, 1, 2, 3]); indices = [0, 1, 2, 3]; del indices[zero_ampl1]
                zero_ampl2 = random.choice(indices)  # select 2nd zero coefficient
                zeroing_coefficients[zero_ampl1] = 0.0; zeroing_coefficients[zero_ampl2] = 0.0
            elif zero_ampls == 3:
                indices = [0, 1, 2, 3]
                zero_ampl1 = random.choice(indices); del indices[zero_ampl1]
                zero_ampl2 = random.choice(indices); del indices[indices.index(zero_ampl2)]  # select 2nd zero coefficient
                zero_ampl3 = random.choice(indices)
                zeroing_coefficients[zero_ampl1] = 0.0; zeroing_coefficients[zero_ampl2] = 0.0; zeroing_coefficients[zero_ampl3] = 0.0
            # Compose random amplitudes in combination
            x1 = zeroing_coefficients[0]*random.random()*x1_max
            x2 = zeroing_coefficients[1]*random.random()*x2_max
            x3 = zeroing_coefficients[2]*random.random()*x3_max
            x4 = zeroing_coefficients[3]*random.random()*x4_max
            x_row = [x1, x2, x3, x4]; x_test.append(x_row)  # add to the data
        x_test = np.asarray(x_test)  # generated test data
        y_test = []
        # Exact test data returned by the function
        for x_row in x_test:
            y = nonlinear_multiparam_model(x_row)
            y_test.append(y)
        y_test = np.asarray(y_test)

        # Fitting test data and compare with the provided by the function
        y_fit = knn_model.predict(x_test)
